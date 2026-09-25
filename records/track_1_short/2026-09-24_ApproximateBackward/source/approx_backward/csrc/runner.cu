// Match the pinned FA3 tile choices for each PR360 head shape.
#include "flash_bwd_launch_template.h"
void launch_head_bwd(Flash_bwd_params &p,cudaStream_t stream) {
    if (p.dv == 128) {
        run_flash_bwd<90,64,64,128,cutlass::bfloat16_t,
            false,true,false,true,false,false,
            2,2,true,false,false,2,1,2,1,false,128>(p,stream);
    } else {
        run_flash_bwd<90,64,128,128,cutlass::bfloat16_t,
            false,true,false,true,false,false,
            2,2,true,false,false,2,1,2,2,false>(p,stream);
    }
}
