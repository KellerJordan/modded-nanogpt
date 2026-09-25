// Parameter construction adapted from FA3 (BSD-3-Clause), pinned source manifest.
#include <torch/types.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include "flash.h"
void launch_head_bwd(Flash_bwd_params &params, cudaStream_t stream);

std::tuple<at::Tensor,at::Tensor,at::Tensor> head_bwd(
    at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor out, at::Tensor lse, at::Tensor cu,
    std::optional<at::Tensor> windows, int64_t maxlen, double scale, int64_t full_window
) {
    TORCH_CHECK(q.is_cuda() && q.dim()==3 && q.scalar_type()==at::kBFloat16);
    c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major==9, "SM90 kernel required");
    int const D=q.size(2), DV=v.size(2);
    TORCH_CHECK(D==64 && (DV==64 || DV==128), "PR360 head shapes only");
    for (auto const& x : {k,v,out,dout}) {
        TORCH_CHECK(x.device()==q.device() && x.scalar_type()==q.scalar_type());
        TORCH_CHECK(x.dim()==3 && x.size(0)==q.size(0) && x.size(1)==q.size(1) && x.stride(2)==1);
    }
    TORCH_CHECK(k.size(2)==D && out.size(2)==DV && dout.size(2)==DV);
    TORCH_CHECK(q.stride(2)==1 && q.size(0)>0 && q.size(1)>0);
    int const n=q.size(0), h=q.size(1), b=cu.numel()-1;
    TORCH_CHECK(cu.device()==q.device() && cu.scalar_type()==at::kInt && cu.dim()==1 && cu.is_contiguous() && b>0);
    TORCH_CHECK(lse.device()==q.device() && lse.scalar_type()==at::kFloat && lse.is_contiguous());
    TORCH_CHECK(lse.dim()==2 && lse.size(0)==h && lse.size(1)==n);
    TORCH_CHECK(maxlen>0 && full_window>=0 && std::isfinite(scale) && scale>0);
    if (windows.has_value()) {
        auto const& w=windows.value();
        TORCH_CHECK(w.device()==q.device() && w.scalar_type()==at::kInt && w.dim()==1 && w.numel()==h && w.is_contiguous());
    }
    auto opts=q.options();
    auto dq=at::empty_like(q),dk=at::empty_like(k),dv=at::empty_like(v);
    int const BM=DV==64?128:64; constexpr int BN=128;
    auto rounded=[](int x,int m){return (x+m-1)/m*m;};
    int const qround=rounded(maxlen,BM),kround=rounded(maxlen,BN);
    int const padded=rounded(n+b*BM,BM);
    auto delta=at::empty({h,padded},opts.dtype(at::kFloat));
    auto lse2=at::empty({h,padded},opts.dtype(at::kFloat));
    auto dq_acc=at::empty({h,padded*D},opts.dtype(at::kFloat));
    auto sem=at::empty({(maxlen+BM-1)/BM,b,h},opts.dtype(at::kInt));
    // This entry point only supports BF16 varlen self-attention: no dropout,
    // GQA, softcap, or paged KV. Unused FlashAttention parameters remain zero.
    Flash_bwd_params p{};
    p.is_bf16 = true;
    p.q_ptr=q.data_ptr(); p.k_ptr=k.data_ptr(); p.v_ptr=v.data_ptr();
    p.o_ptr=out.data_ptr(); p.do_ptr=dout.data_ptr();
    p.dq_ptr=dq.data_ptr(); p.dk_ptr=dk.data_ptr(); p.dv_ptr=dv.data_ptr();
    p.q_row_stride=q.stride(0); p.k_row_stride=k.stride(0); p.v_row_stride=v.stride(0);
    p.o_row_stride=out.stride(0); p.do_row_stride=dout.stride(0);
    p.dq_row_stride=dq.stride(0); p.dk_row_stride=dk.stride(0); p.dv_row_stride=dv.stride(0);
    p.q_head_stride=q.stride(1); p.k_head_stride=k.stride(1); p.v_head_stride=v.stride(1);
    p.o_head_stride=out.stride(1); p.do_head_stride=dout.stride(1);
    p.dq_head_stride=dq.stride(1); p.dk_head_stride=dk.stride(1); p.dv_head_stride=dv.stride(1);
    p.v_dim_stride=v.stride(2);
    p.cu_seqlens_q=cu.data_ptr<int>(); p.cu_seqlens_k=cu.data_ptr<int>();
    p.softmax_lse_ptr=lse.data_ptr(); p.dsoftmax_sum=delta.data_ptr();
    p.dq_accum_ptr=dq_acc.data_ptr();
    p.b=b; p.h=h; p.h_k=h;
    p.seqlen_q=maxlen; p.seqlen_k=maxlen;
    p.seqlen_q_rounded=qround; p.seqlen_k_rounded=kround;
    p.d=D; p.d_rounded=D;
    p.scale_softmax=scale;
    p.p_dropout=1.0f; p.rp_dropout=1.0f; p.p_dropout_in_uint8_t=255;
    p.is_local=true; p.window_size_left=full_window;
    p.arch=90; p.num_sm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    p.total_q=n;p.total_k=n;p.softmax_lse_log2_ptr=lse2.data_ptr();p.dv=DV;p.dv_rounded=DV;
    p.dq_semaphore=sem.data_ptr<int>();
    p.head_windows_left=windows.has_value() ? windows.value().data_ptr<int>() : nullptr;
    launch_head_bwd(p,at::cuda::getCurrentCUDAStream().stream());
    return {dq,dk,dv};
}

#define HEAD_LIBRARY_EXPAND(ns,m) TORCH_LIBRARY(ns,m)
#define HEAD_LIBRARY_IMPL_EXPAND(ns,key,m) TORCH_LIBRARY_IMPL(ns,key,m)
HEAD_LIBRARY_EXPAND(TORCH_EXTENSION_NAME,m) {
    m.def("bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor cu, Tensor? windows, int maxlen, float scale, int full_window) -> (Tensor, Tensor, Tensor)");
}
HEAD_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME,CUDA,m) {
    m.impl("bwd",&head_bwd);
}
