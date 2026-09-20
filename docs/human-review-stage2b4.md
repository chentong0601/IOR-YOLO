# Stage 2B-4H 人工复核表

本表记录21个指定案例的用户人工裁决。自动证据及图像见 [human_review.html](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html)。下方图像证据和反事实影响表为**裁决前快照**；逐案 Human Decision 才是当前决定。仅 C01/C02 合组及 C01/C02/B01/B02 的关系置信度据此更新。未建立正式 split。

| Case | 对象 | 裁决前状态 | 裁决前反事实 Source-group 影响 |
|---|---|---|---|
| [C01](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#C01) | img_13960 | Candidate | Current: two separate source groups sg-0583 and sg-1080 (1+1 original images). Accept relation: one 2-image group, total group count 1114→1113. Reject/keep Candidate: current structure unchanged. |
| [C02](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#C02) | img_14040 | Candidate | Current: two separate source groups sg-0588 and sg-1081 (1+1 original images). Accept relation: one 2-image group, total group count 1114→1113. Reject/keep Candidate: current structure unchanged. |
| [B01](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#B01) | test/3100.jpg ↔ train/1640.jpg | Supported | Accept: 2 original images remain one group sg-0046. Reject: split into Group A (1: test/3100.jpg) and Group B (1: train/1640.jpg); total group count 1114→1115. |
| [B02](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#B02) | test/1270.jpg ↔ train/2340.jpg | Supported | Accept: 4 original images remain one group sg-0006. Reject: split into Group A (1: test/1270.jpg) and Group B (3: train/2340.jpg, train/2340_brightness.jpg, val/2340_noise.jpg); total group count 1114→1115. |
| [S01](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S01) | 1770 | Strongly Supported | Current: 3 original images in sg-0288. Reject this weakest edge: no source-group change because another supported path remains. |
| [S02](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S02) | img_13700 | Strongly Supported | Current: 2 original images in sg-0570. Reject this weakest edge: component splits into sizes 1+1; group count +1. |
| [S03](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S03) | 2040 | Strongly Supported | Current: 3 original images in sg-0023. Reject this weakest edge: no source-group change because another supported path remains. |
| [S04](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S04) | 194 | Strongly Supported | Current: 3 original images in sg-0303. Reject this weakest edge: no source-group change because another supported path remains. |
| [S05](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S05) | img_1406 | Strongly Supported | Current: 3 original images in sg-0589. Reject this weakest edge: no source-group change because another supported path remains. |
| [S06](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S06) | img_13650 | Strongly Supported | Current: 3 original images in sg-0094. Reject this weakest edge: no source-group change because another supported path remains. |
| [S07](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S07) | 1180 | Strongly Supported | Current: 2 original images in sg-0249. Reject this weakest edge: component splits into sizes 1+1; group count +1. |
| [S08](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S08) | 257 | Strongly Supported | Current: 6 original images in sg-0035. Reject this weakest edge: no source-group change because another supported path remains. |
| [S09](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S09) | 1550 | Strongly Supported | Current: 3 original images in sg-0274. Reject this weakest edge: no source-group change because another supported path remains. |
| [S10](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#S10) | img_1381 | Strongly Supported | Current: 2 original images in sg-0577. Reject this weakest edge: component splits into sizes 1+1; group count +1. |
| [D01](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#D01) | test/138.jpg ↔ train/251.jpg | Confirmed image duplicate; annotation conflict Unresolved | Identical bytes require the same source group sg-0008 (6 original images). Counterfactually removing only this edge yields 1 component(s) with sizes 6; annotation choice does not change the duplicate fact. |
| [D02](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#D02) | train/138_brightness.jpg ↔ val/251_brightness.jpg | Confirmed image duplicate; annotation conflict Unresolved | Identical bytes require the same source group sg-0008 (6 original images). Counterfactually removing only this edge yields 1 component(s) with sizes 6; annotation choice does not change the duplicate fact. |
| [D03](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#D03) | train/398.jpg ↔ val/166.jpg | Confirmed image duplicate; annotation conflict Unresolved | Identical bytes require the same source group sg-0419 (2 original images). Counterfactually removing only this edge yields 2 component(s) with sizes 1+1; annotation choice does not change the duplicate fact. |
| [R01](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#R01) | train/172_brightness.jpg | Unresolved | Both files are provisionally attached to source group sg-0285 by filename. Accept/reject pixel equivalence does not change the 1114 original-image groups; it changes whether this resize representation may be used. |
| [R02](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#R02) | val/172_noise.jpg | Unresolved | Both files are provisionally attached to source group sg-0285 by filename. Accept/reject pixel equivalence does not change the 1114 original-image groups; it changes whether this resize representation may be used. |
| [R03](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#R03) | train/327.jpg | Unresolved | Both files are provisionally attached to source group sg-0389 by filename. Accept/reject pixel equivalence does not change the 1114 original-image groups; it changes whether this resize representation may be used. |
| [U01](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#U01) | test/IMG_54350.jpg #3 | Unlabeled / Unknown | No source-group change (sg-0148). Human decision affects derived annotation and whether this image may enter scored evaluation. |

以下勾选和文字记录用户已作出的决定。表格中的旧 source-group ID 与 1114 基数仅用于复核裁决前的影响；当前重建结果为 **1112 groups**，最新组号以重新生成的 source_groups.csv 为准。Raw Data 未编辑。

## C01 — img_13960

- 类型：Candidate family；裁决前 confidence：Candidate；relation type：same_annotation_geometry / filename_family。
- 关键文件：train/IMG_13960_brightness.jpg ↔ val/IMG_13960.jpg。
- 裁决前 Source-group 反事实影响：Current: two separate source groups sg-0583 and sg-1080 (1+1 original images). Accept relation: one 2-image group, total group count 1114→1113. Reject/keep Candidate: current structure unchanged.
- 证据与薄弱点：Same polygon list and augmentation-like suffix, but no verified capture ID; do not upgrade automatically.
- Human Decision: **Accept relation; merge source groups; Strongly Supported.** 这是用户的人工同源裁决，不宣称 Confirmed physical acquisition。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## C02 — img_14040

- 类型：Candidate family；裁决前 confidence：Candidate；relation type：same_annotation_geometry / filename_family。
- 关键文件：train/IMG_14040_brightness.jpg ↔ val/IMG_14040.jpg。
- 裁决前 Source-group 反事实影响：Current: two separate source groups sg-0588 and sg-1081 (1+1 original images). Accept relation: one 2-image group, total group count 1114→1113. Reject/keep Candidate: current structure unchanged.
- 证据与薄弱点：Same polygon list and augmentation-like suffix, but no verified capture ID; do not upgrade automatically.
- Human Decision: **Accept relation; merge source groups; Strongly Supported.** 这是用户的人工同源裁决，不宣称 Confirmed physical acquisition。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## B01 — test/3100.jpg ↔ train/1640.jpg

- 类型：Supported bridge；裁决前 confidence：Supported；relation type：high_visual_similarity。
- 关键文件：test/3100.jpg ↔ train/1640.jpg。
- 裁决前 Source-group 反事实影响：Accept: 2 original images remain one group sg-0046. Reject: split into Group A (1: test/3100.jpg) and Group B (1: train/1640.jpg); total group count 1114→1115.
- 证据与薄弱点：This edge is a graph bridge: removing it disconnects the current component. Human review must assess source identity, not just visual similarity.
- Human Decision: **Accept relation; keep the existing entire component together; human-reviewed Strongly Supported.** 不宣称已取得采集 ID。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## B02 — test/1270.jpg ↔ train/2340.jpg

- 类型：Supported bridge；裁决前 confidence：Supported；relation type：high_visual_similarity。
- 关键文件：test/1270.jpg ↔ train/2340.jpg。
- 裁决前 Source-group 反事实影响：Accept: 4 original images remain one group sg-0006. Reject: split into Group A (1: test/1270.jpg) and Group B (3: train/2340.jpg, train/2340_brightness.jpg, val/2340_noise.jpg); total group count 1114→1115.
- 证据与薄弱点：This edge is a graph bridge: removing it disconnects the current component. Human review must assess source identity, not just visual similarity.
- Human Decision: **Accept relation; keep the existing entire component together; human-reviewed Strongly Supported.** 不宣称已取得采集 ID。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S01 — 1770

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/1770_brightness.jpg ↔ val/1770_noise.jpg。
- 裁决前 Source-group 反事实影响：Current: 3 original images in sg-0288. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.995345 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S02 — img_13700

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/IMG_13700_brightness.jpg ↔ val/IMG_13700.jpg。
- 裁决前 Source-group 反事实影响：Current: 2 original images in sg-0570. Reject this weakest edge: component splits into sizes 1+1; group count +1.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.995642 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S03 — 2040

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/2040_noise.jpg ↔ val/2040_brightness.jpg。
- 裁决前 Source-group 反事实影响：Current: 3 original images in sg-0023. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.995814 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S04 — 194

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/194_brightness.jpg ↔ val/194_noise.jpg。
- 裁决前 Source-group 反事实影响：Current: 3 original images in sg-0303. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.995840 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S05 — img_1406

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/IMG_1406.jpg ↔ val/IMG_1406_brightness.jpg。
- 裁决前 Source-group 反事实影响：Current: 3 original images in sg-0589. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.996001 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S06 — img_13650

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：test/IMG_13650.jpg ↔ train/IMG_13650_brightness.jpg。
- 裁决前 Source-group 反事实影响：Current: 3 original images in sg-0094. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.996408 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S07 — 1180

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/1180_brightness.jpg ↔ val/1180.jpg。
- 裁决前 Source-group 反事实影响：Current: 2 original images in sg-0249. Reject this weakest edge: component splits into sizes 1+1; group count +1.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.996412 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S08 — 257

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：test/257.jpg ↔ train/257_brightness.jpg。
- 裁决前 Source-group 反事实影响：Current: 6 original images in sg-0035. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.996486 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S09 — 1550

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/1550_brightness.jpg ↔ val/1550_noise.jpg。
- 裁决前 Source-group 反事实影响：Current: 3 original images in sg-0274. Reject this weakest edge: no source-group change because another supported path remains.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.996601 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## S10 — img_1381

- 类型：Weakest Strongly Supported；裁决前 confidence：Strongly Supported；relation type：augmentation_derivative。
- 关键文件：train/IMG_1381_brightness.jpg ↔ val/IMG_1381.jpg。
- 裁决前 Source-group 反事实影响：Current: 2 original images in sg-0577. Reject this weakest edge: component splits into sizes 1+1; group count +1.
- 证据与薄弱点：Current rule: terminal augmentation family + identical region list + 64×64 correlation 0.996648 ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified.
- Human Decision: **Accept relation; retain existing source-group membership and Strongly Supported.** 缺少 acquisition ID，不升级为 Confirmed。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## D01 — test/138.jpg ↔ train/251.jpg

- 类型：Exact SHA duplicate / annotation conflict；裁决前 confidence：Confirmed image duplicate; annotation conflict Unresolved；relation type：exact_duplicate。
- 关键文件：test/138.jpg ↔ train/251.jpg。
- 裁决前 Source-group 反事实影响：Identical bytes require the same source group sg-0008 (6 original images). Counterfactually removing only this edge yields 1 component(s) with sizes 6; annotation choice does not change the duplicate fact.
- 证据与薄弱点：Region counts: 1 vs 1; class lists equal=True; polygon coordinates differ=True; union-mask IoU=0.9918 (rasterized, not instance-matched). Cause of reannotation unknown.
- Human Decision: **Accept image-duplicate relation (Confirmed); identical-SHA files remain in one source group.** Polygon 有轻微差异，annotation conflict 仍为 Unresolved；不平均、不修复、不选择一份为真值。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## D02 — train/138_brightness.jpg ↔ val/251_brightness.jpg

- 类型：Exact SHA duplicate / annotation conflict；裁决前 confidence：Confirmed image duplicate; annotation conflict Unresolved；relation type：exact_duplicate。
- 关键文件：train/138_brightness.jpg ↔ val/251_brightness.jpg。
- 裁决前 Source-group 反事实影响：Identical bytes require the same source group sg-0008 (6 original images). Counterfactually removing only this edge yields 1 component(s) with sizes 6; annotation choice does not change the duplicate fact.
- 证据与薄弱点：Region counts: 1 vs 1; class lists equal=True; polygon coordinates differ=True; union-mask IoU=0.9918 (rasterized, not instance-matched). Cause of reannotation unknown.
- Human Decision: **Accept image-duplicate relation (Confirmed); identical-SHA files remain in one source group.** Polygon 有轻微差异，annotation conflict 仍为 Unresolved；不平均、不修复、不选择一份为真值。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## D03 — train/398.jpg ↔ val/166.jpg

- 类型：Exact SHA duplicate / annotation conflict；裁决前 confidence：Confirmed image duplicate; annotation conflict Unresolved；relation type：exact_duplicate。
- 关键文件：train/398.jpg ↔ val/166.jpg。
- 裁决前 Source-group 反事实影响：Identical bytes require the same source group sg-0419 (2 original images). Counterfactually removing only this edge yields 2 component(s) with sizes 1+1; annotation choice does not change the duplicate fact.
- 证据与薄弱点：Region counts: 2 vs 2; class lists equal=True; polygon coordinates differ=True; union-mask IoU=0.9885 (rasterized, not instance-matched). Cause of reannotation unknown.
- Human Decision: **Accept image-duplicate relation (Confirmed); identical-SHA files remain in one source group.** Polygon 有轻微差异，annotation conflict 仍为 Unresolved；不平均、不修复、不选择一份为真值。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## R01 — train/172_brightness.jpg

- 类型：Abnormal original ↔ resize；裁决前 confidence：Unresolved；relation type：unresolved_candidate / same-name resize companion。
- 关键文件：train/172_brightness.jpg ↔ train/172_brightness.jpg。
- 裁决前 Source-group 反事实影响：Both files are provisionally attached to source group sg-0285 by filename. Accept/reject pixel equivalence does not change the 1114 original-image groups; it changes whether this resize representation may be used.
- 证据与薄弱点：LANCZOS RGB MAE=39.118435; scaled polygon mean/max residual=7.292861/27.656442 px. Pairing/processing cause remains unresolved.
- Human Decision: **Accept same-source companion; deterministic resize equivalence Unresolved.** 正式实验排除 dataset-provided resize 表示；输入尺寸由 training pipeline 动态 resize；Raw Data 不变。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## R02 — val/172_noise.jpg

- 类型：Abnormal original ↔ resize；裁决前 confidence：Unresolved；relation type：unresolved_candidate / same-name resize companion。
- 关键文件：val/172_noise.jpg ↔ val/172_noise.jpg。
- 裁决前 Source-group 反事实影响：Both files are provisionally attached to source group sg-0285 by filename. Accept/reject pixel equivalence does not change the 1114 original-image groups; it changes whether this resize representation may be used.
- 证据与薄弱点：LANCZOS RGB MAE=36.476682; scaled polygon mean/max residual=7.292861/27.656442 px. Pairing/processing cause remains unresolved.
- Human Decision: **Accept same-source companion; deterministic resize equivalence Unresolved.** 正式实验排除 dataset-provided resize 表示；输入尺寸由 training pipeline 动态 resize；Raw Data 不变。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## R03 — train/327.jpg

- 类型：Abnormal original ↔ resize；裁决前 confidence：Unresolved；relation type：unresolved_candidate / same-name resize companion。
- 关键文件：train/327.jpg ↔ train/327.jpg。
- 裁决前 Source-group 反事实影响：Both files are provisionally attached to source group sg-0389 by filename. Accept/reject pixel equivalence does not change the 1114 original-image groups; it changes whether this resize representation may be used.
- 证据与薄弱点：LANCZOS RGB MAE=31.310911; scaled polygon mean/max residual=2.147004/6.425076 px. Pairing/processing cause remains unresolved.
- Human Decision: **Accept same-source companion; deterministic resize equivalence Unresolved.** 正式实验排除 dataset-provided resize 表示；输入尺寸由 training pipeline 动态 resize；Raw Data 不变。
- [x] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。

## U01 — test/IMG_54350.jpg #3

- 类型：Unlabeled polygon；裁决前 confidence：Unlabeled / Unknown；relation type：annotation integrity issue。
- 关键文件：test/IMG_54350.jpg ↔ test/IMG_54350.jpg。
- 裁决前 Source-group 反事实影响：No source-group change (sg-0148). Human decision affects derived annotation and whether this image may enter scored evaluation.
- 证据与薄弱点：Four regions: #0 mature, #1 semi-mature, #2 mature, #3 Unknown. Polygon #3 has geometry but no class attribute; do not infer its maturity stage.
- Human Decision: **Invalid / unknown region.** 不赋 maturity class；派生三类监督标注不将 #3 当作有效训练或评价目标；保留本图及其余三个有效 labeled regions；raw annotation 不变。
- [x] Exclude only invalid / unknown region from derived three-class targets; retain image
- [ ] Accept relation
- [ ] Reject relation
- [ ] Keep Candidate
- [ ] Exclude from evaluation
- [ ] Needs more evidence
- Reason: 用户在本轮明确给出的人工决定；证据图像及原先风险见本案例和 HTML 审核页。
