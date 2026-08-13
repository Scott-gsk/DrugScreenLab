# Progress

## Final execution checkpoint (2026-08-13)

- Official XPert source commit and public Figshare assets are registered in `mvp/foundation/xpert/ASSET_MANIFEST.json`; large assets remain external/ignored.
- Official demo, warm checkpoint sanity, full L1000 `split_cold_cell_1`, and full L1000 `split_cold_drug_1` bounded runs completed through the official loader/model/loss/evaluation path.
- Authoritative independent test profiles are recorded in `mvp/foundation/xpert/XPERT_FOUNDATION_RESULT.json`.
- Cold-cell Delta978 Pearson=0.282 and cold-drug Delta978 Pearson=0.356; both prediction profiles are finite and non-collapsed.
- XPert is marked `XPERT_FOUNDATION_READY`; MultiDCP fallback was not activated.
- Exact978 -> disease reversal -> Broad PRISM adapter is integrated and recorded in `mvp/foundation/xpert/ADAPTER_DOWNSTREAM_RESULT.json`.
- `EXP-004` remains untouched; `EXP-005` remains reserved for the first Novel Extension.

## 2026-08-13

- 已读取 `AGENTS.md`、Master Plan、`PROJECT_STATE.yaml`、`experiments/registry.yaml`。
- 已激活 brainstorming、using-superpowers、planning-with-files-zh；当前处于设计门槛，未实施 XPert。
- 已通过官方 XPert GitHub、Nature Machine Intelligence 页面、MultiDCP GitHub 做 primary-source verification。
- 用户已明确：不创建新 EXP；XPert Reset 作为独立 FOUNDATION TRACK 立即执行；EXP-005 保留给首个 Novel Extension。
- 用户已明确授权普通环境/依赖/路径/数据格式兼容修复、bounded cold-cell/cold-drug reproduction、MultiDCP fallback，以及 XPert Ready 后的 PRISM 接入。
