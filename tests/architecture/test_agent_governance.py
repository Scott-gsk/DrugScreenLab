from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "AGENTS.md"
AGENT_DIR = ROOT / ".codex" / "agents"


def test_agent_rules_point_to_authoritative_state_and_governance():
    text = RULES.read_text()
    # 用户于2026-09-24明确取消机械字数上限，按实际约束维护入口。
    assert "docs/AGENT_GOVERNANCE.md" in text
    for path in (
        "PROJECT_STATE.yaml",
        "experiments/registry.yaml",
        "data/registry/datasets.json",
        "experiments/records/EXP-###.md",
    ):
        assert path in text


def test_agent_rules_do_not_hardcode_the_current_research_track():
    text = RULES.read_text()
    assert "不写死当前研究方向、模型、数据集或 EXP 编号" in text
    assert "breast_pdo_transfer" not in text
    assert "EXP-010" not in text
    assert "PharmaFormer" not in text
    assert "根据状态文件确认当前主线" in text


def test_agent_rules_protect_data_and_evaluation():
    text = RULES.read_text()
    assert "data/raw/" in text
    assert "checksum" in text
    assert "禁止用测试结果选择样本、特征或阈值" in text
    assert "大型矩阵、原始数据、checkpoint、预测文件和日志不进入 Git" in text


def test_agent_rules_fix_the_runtime_and_validation_commands():
    text = RULES.read_text()
    assert "WSL2 Conda `drugscreening-gpu`" in text
    assert "python -m pytest --capture=no" in text
    assert "python -m drug_screen.data.registry --root data" in text
    assert "不得使用 Windows Python" in text


def test_agent_rules_keep_research_and_engineering_distinct():
    text = RULES.read_text()
    assert "工程整理、修复和测试不创建 EXP" in text
    assert "研究工作必须登记 EXP" in text
    assert "开始训练或正式评测前" in text
    assert "获得用户批准" in text
    assert "不把准备工作写成实验完成" in text


def test_agent_rules_forbid_destructive_git_cleanup():
    text = RULES.read_text()
    assert "git clean" in text
    assert "git reset --hard" in text
    assert "不改写 master" in text
    assert "删除前确认路径、引用和恢复点" in text


def test_agent_rules_define_orchestration_and_current_user_configuration():
    text = RULES.read_text()
    assert "复杂研究采用先进模型" in text
    assert "可由 Manager 调用子 Agent" in text
    assert "用户当前配置" in text
    assert "Manager 必须复核执行方（子 Agent）" in text


def test_optional_agent_templates_remain_well_formed():
    expected = {
        "research_manager.toml": "research_manager",
        "scientific_analyst.toml": "scientific_analyst",
        "data_bioinformatics_steward.toml": "data_bioinformatics_steward",
        "model_engineer.toml": "model_engineer",
        "evaluation_statistics_analyst.toml": "evaluation_statistics_analyst",
        "reviewer.toml": "independent_reviewer",
    }
    templates = {path.name: tomllib.loads(path.read_text()) for path in AGENT_DIR.glob("*.toml")}
    assert {name: template["name"] for name, template in templates.items()} == expected
    assert templates["reviewer.toml"]["sandbox_mode"] == "read-only"
    for template in templates.values():
        assert template["model"] == "gpt-6-luna"
        assert template["model_reasoning_effort"] == "max"
        assert "docs/AGENT_GOVERNANCE.md" in template["developer_instructions"]
