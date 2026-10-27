# 결정적 실험 (검토 의견 1순위, 3·4순위 대조군 포함)

같은 파이프라인에서 similarity map만 바꾸거나 선택 규칙만 바꾼 조건을 비교하여, LPIPS 개선이 DINOv2 신호에서 오는지, 선택 구조나 가우시안 크기에서 오는지를 분리한다.

## 실험군

1차 실험군은 다음과 같다. 모든 조건은 gradient 집합과의 합집합, ADC clone/split/prune 규칙(`--sem_split adc`)을 공유한다.

- `dino_l03`: DINOv2, λ_s = 0.3 (제안 방법). 기준 조건이며, SSIM을 세 번째 보조 채널로 함께 역투영하여 두 점수의 상관만 기록한다.
- `ssim_l03`, `l1_l03`: S_π를 SSIM map, 1−L1 map으로 교체한다. semantic 신호가 픽셀 오차보다 나은지 본다.
- `dino_l0`, `ssim_l0`, `l1_l0`: λ_s = 0. scale weight가 선택을 지배하는지 본다(리뷰 4순위).
- `rand_matched`: 층화 샘플링과 같은 개수를 유효 후보에서 무작위로 고른다. 가우시안을 더 넣은 효과와 구분한다(리뷰 3순위 b).
- `radius_matched`: 같은 개수를 투영 반지름 상위에서 고른다. 큰 가우시안 split(Mini-Splatting의 blur split)과 구분한다.

2차(선택) 실험군은 `l1blur_l03`(공간 평활 수준을 맞춘 L1), `dino_viewmin`(view 평균 대신 최악 view), `r9_ssim`·`r9_dino`([9]의 가중 합·view 최대·임계값 0.1·단계당 5% 상한, 재구현), `r9_dino_mean`(가중 합 대신 가중 평균)이다.

## 실행

```bash
# 1) CPU 검증 (GPU 불필요, 기존 커밋 343c470과 densification 결과가 같은지 확인)
python tests/test_decisive_cpu.py

# 2) 짧은 동작 확인 (densification 600~1100 구간만)
python train_sim.py -s datasets/NeRF360/flowers -i images_4 -m output/smoke --eval --disable_viewer \
  --iterations 1500 --densify_until_iter 1200 --test_iterations -1 \
  --densify_grad_threshold 0.0002 --sem_sample_ratio 0.3 --sem_log --sem_signal dino --sem_shadow ssim

# 3) 파일럿 (flowers, seed 0)
SCENES="flowers" SEEDS="0" RHO=0.3 bash run_decisive.sh

# 4) 본 실험 / 결과 취합
SCENES="flowers treehill garden truck" SEEDS="0 1 2" RHO=0.3 bash run_decisive.sh
python collect_results.py --root output/decisive
```

데이터 경로는 `M360_DIR`, `TANDT_DIR`, `DB_DIR`, 출력 경로는 `OUT_ROOT` 환경변수로 바꾼다. DINOv2를 쓰는 조건은 저장소 최상위에 `dinov2/`가 있어야 한다.

**주의.** `RHO`는 보고된 Ours를 만든 ρ 값으로 맞춰야 한다. 커밋 343c470의 코드에는 0.2, 논문에는 0.3으로 적혀 있다. 임계값 τ_g는 스크립트가 0.0002를 명시적으로 넘기므로 `arguments/__init__.py`의 기본값과 무관하다.

## 출력 (`output/decisive/<scene>/<arm>/seed<k>/`)

- `sem_run_config.json`: 전체 옵션, 해석된 설정, seed, git 커밋
- `train_stats.json`: 학습 시간(테스트·저장 시간 포함), densification 구간 시간, peak VRAM(PyTorch allocator 기준), 최종 #GS
- `sem_log.jsonl`: densification 단계별 |M_sem|, |M_grad|, 교집합, semantic 단독 수, 구간별 c_m·K_m, clone/split 수, M_sem과 전체 유효 후보의 투영 반지름 분포(R_max 이상 비율), M_sem의 gradient가 τ_g 미만인 비율, clamp 전 음수 점수 비율, shadow 신호와의 Spearman 상관, opacity reset 이후 경과 iteration, [9] 모드의 상한 도달 여부
- `sem_snapshots/iter_*.npz`: 1000·3000·6000·9000·12000·14900 근처 단계의 가우시안별 점수, gradient, 투영 반지름, 선택 여부 (S_k–ḡ_k 산점도용, 최대 20만 개)
- `results.json`, `per_view.json`: 기존 `metrics.py` 결과
- `eval_maps.json`: 테스트 view 렌더의 DINOv2·SSIM·L1 map 평균, 하위 5%·10% 분위수, 고정 임계값 아래 픽셀 비율
- `output/decisive/_summary/runs.csv`, `summary.csv`: 실행별 결과와 seed 평균±표준편차, 장면 평균(`scene=ALL`)

## 새 옵션 (`arguments/__init__.py`)

`--sem_signal dino|ssim|l1|l1_blur`, `--sem_mode ours|r9`, `--sem_select stratified|random_matched|radius_matched`, `--sem_view_norm auto|mean|sum`, `--sem_view_agg auto|mean|min|max`, `--sem_lambda_scale`, `--sem_radius_min`, `--sem_radius_max`, `--sem_sample_ratio`, `--sem_r9_threshold`, `--sem_r9_cap`, `--sem_blur_sigma`, `--sem_shadow`, `--sem_split adc|mask2`, `--sem_log`, 그리고 `train_sim.py`의 `--seed` (0이면 기존과 같은 고정 seed).

기본값(dino, ours, stratified, adc, shadow 없음)에서는 기존 계산과 같다. ours 모드에서 가중 합(`--sem_view_norm sum`)은 막아 두었다. goodness 점수를 합하면 넓게 덮는 가우시안일수록 잘 복원된 것으로 판정되기 때문이며, 가중 합 비교는 오차 점수를 쓰는 r9 모드에서 한다.
