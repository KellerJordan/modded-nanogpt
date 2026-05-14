# Dampen radial gradient component

This submission builds on PR291's Contra-Muon to Soft-Muon setup and adds
post-step radial damping. After applying the optimizer update, the hidden matrix
parameter norm is rescaled so outward radial movement is dampened while inward
movement is preserved according to the constants logged in each run.

The submitted step count is **2990**. The logs include a fixed validation grid
at 2990, 2995, 2999, 3000, 3010, and 3020 for all seeds.

Across 11 non-cherry-picked seed logfiles, the step 2990 mean validation loss is
3.27866818. Under the Track 3 README criterion,
`(3.28 - mu) * sqrt(n) = 0.00441714`, which exceeds the required `0.004`
threshold. Equivalently, using sigma=0.0013 gives `z=3.3978` and one-sided
`p=0.00034`, satisfying the p<0.001 criterion at 2990 steps.

| Seed | Log | 2990 val | 2995 val | 2999 val | 3000 val | 3010 val | 3020 val |
| -: | - | -: | -: | -: | -: | -: | -: |
| 0 | [d0110280-d17c-47a7-b5bf-79f5900d5711.txt](d0110280-d17c-47a7-b5bf-79f5900d5711.txt) | 3.27963 | 3.27931 | 3.27907 | 3.27900 | 3.27832 | 3.27766 |
| 1 | [d2e400ab-629a-48d9-ae1d-9cd5767006d2.txt](d2e400ab-629a-48d9-ae1d-9cd5767006d2.txt) | 3.27996 | 3.27967 | 3.27942 | 3.27935 | 3.27866 | 3.27800 |
| 2 | [a4be63d8-39f8-43a8-ab87-f3c7e705f156.txt](a4be63d8-39f8-43a8-ab87-f3c7e705f156.txt) | 3.27794 | 3.27762 | 3.27739 | 3.27733 | 3.27664 | 3.27600 |
| 3 | [1fafaa5c-cfbb-40f7-a19c-791cb14454e7.txt](1fafaa5c-cfbb-40f7-a19c-791cb14454e7.txt) | 3.27908 | 3.27879 | 3.27853 | 3.27847 | 3.27777 | 3.27711 |
| 4 | [fcf7a0d1-fdb6-4856-a7c0-3c762f201f68.txt](fcf7a0d1-fdb6-4856-a7c0-3c762f201f68.txt) | 3.28073 | 3.28049 | 3.28024 | 3.28018 | 3.27944 | 3.27880 |
| 5 | [00882c75-914d-4340-8e0b-dcffcb18b73d.txt](00882c75-914d-4340-8e0b-dcffcb18b73d.txt) | 3.27822 | 3.27792 | 3.27771 | 3.27766 | 3.27698 | 3.27633 |
| 6 | [816f035f-c55d-4a67-9c1f-855d4d34cf5c.txt](816f035f-c55d-4a67-9c1f-855d4d34cf5c.txt) | 3.27770 | 3.27739 | 3.27717 | 3.27710 | 3.27641 | 3.27573 |
| 7 | [5ba99921-2f46-413c-926a-483e854b4471.txt](5ba99921-2f46-413c-926a-483e854b4471.txt) | 3.27743 | 3.27712 | 3.27689 | 3.27683 | 3.27610 | 3.27548 |
| 8 | [5770362e-d11c-4fae-a30d-be3e19c18eef.txt](5770362e-d11c-4fae-a30d-be3e19c18eef.txt) | 3.27824 | 3.27796 | 3.27772 | 3.27765 | 3.27695 | 3.27630 |
| 9 | [e6232c5e-4797-485c-8fba-11630083685d.txt](e6232c5e-4797-485c-8fba-11630083685d.txt) | 3.28000 | 3.27973 | 3.27951 | 3.27944 | 3.27871 | 3.27806 |
| 10 | [52f5a74c-6e78-4104-8f87-162da37c4933.txt](52f5a74c-6e78-4104-8f87-162da37c4933.txt) | 3.27642 | 3.27612 | 3.27585 | 3.27578 | 3.27507 | 3.27443 |
| **Mean** |  | **3.27866818** | **3.27837455** | **3.27813636** | **3.27807182** | **3.27736818** | **3.27671818** |

## Credits

This result descends from PR291 and the same prior submissions credited there:
PR274 Skylight-001, PR275 Contra-Muon, PR278 MLP SOAP preconditioning, PR283
Trustlight, and PR287 power-law LR scheduling.
