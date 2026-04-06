from llm_agent.agent import AgentConfig, OrchestratedAgent
from llm_agent.model_clients import MockClient


def test_generic_symbolic_variant_disables_benchmark_specific_rescue() -> None:
    question = (
        "Adrien's total salary was 30 percent higher than Lylah's salary four years ago. "
        "Adrien earned $40000 four years ago. They each earn 40% more now. "
        "What is their combined salary now?"
    )
    full_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(mode="direct", use_symbolic_solver=True, use_query_rewrite=False),
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    full_pred, full_trace = full_agent.solve(question)
    generic_pred, generic_trace = generic_agent.solve(question)

    assert full_pred == "95200"
    assert full_trace["symbolic_solver_used"] is True
    assert generic_pred != "95200"
    assert generic_trace["mode"] == "direct"
    assert "symbolic_solver_used" not in generic_trace


def test_generic_symbolic_variant_keeps_iid_derived_rule() -> None:
    # Sum+difference schema derived from IID questions only (coins, football).
    # "110 total, 30 more gold than silver. How many gold coins?" => 70
    question = (
        "Gretchen has 110 coins. There are 30 more gold coins than silver coins. "
        "How many gold coins does Gretchen have?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "70"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_reverse_discount() -> None:
    question = (
        "A book costs $19.50 with a 25% discount from the original price. "
        "What was the original price?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "26"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_percent_of_remainder() -> None:
    question = (
        "In a class of 20 students, 20% enrolled in one dance, 25% of the remaining "
        "enrolled in another dance, and the rest chose a third dance. "
        "What percentage chose the third dance?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "60"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_clock_duration_rate() -> None:
    question = (
        "A candle melts by 2 centimeters every hour that it burns. "
        "How many centimeters shorter will it be from 1:00 PM to 5:00 PM?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "8"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_installment_interest() -> None:
    question = (
        "A buyer purchased five phones for $150 each on a 3-month installment. "
        "A 2% interest is charged for each unit. How much is paid each month?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "255"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_story_inventory_math() -> None:
    question = (
        "Charlie had 10 stickers. He bought 21 stickers and got 23 stickers for his birthday. "
        "Then Charlie gave 9 stickers to his sister and used 28 stickers to decorate a card. "
        "How many stickers does Charlie have now?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "17"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_per_unit_need_cost() -> None:
    question = (
        "A host needs 0.75 gift bags per invited guest. She invited 16 friends. "
        "Gift bags are $2 each. How much will she spend?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "24"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_adaptive_router_with_tools_uses_balanced_module_selection() -> None:
    question = (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?"
    )
    agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="adaptive_router",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
            routing_fast_k=3,
            routing_conf_threshold=0.67,
            use_verifier=True,
        ),
    )

    pred, trace = agent.solve(question)

    assert pred == "3"
    assert trace["mode"] == "adaptive_router"
    assert trace["route"] == "balanced_modules"
    assert trace["selected_module"] == "symbolic"
    assert trace["used_symbolic_candidate"] is True
    assert "candidate_scores" in trace


def test_generic_symbolic_variant_handles_rounded_multi_item_revenue() -> None:
    question = (
        "A seller has three items priced at $2.74, $1.87, and $2.12 per pot. "
        "All prices are rounded to the nearest dollar. If she sells 12, 9, and 17 pots, "
        "how much does she make?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "88"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_return_trip_with_idle_and_segments() -> None:
    question = (
        "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he "
        "forgot something very important at home. He tries to get home in 4 hours but spends the first "
        "2 hours in standstill traffic. He spends the next half-hour driving at a speed of 30mph, "
        "before being able to drive the remaining time of the 4 hours going at 80 mph. "
        "How far is he from home at the end of those 4 hours?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )

    pred, trace = generic_agent.solve(question)

    assert pred == "45"
    assert trace["symbolic_solver_used"] is True
    assert trace["symbolic_solver_variant"] == "generic"


def test_generic_symbolic_variant_handles_bundle_best_price_savings() -> None:
    question = (
        "Vincent can buy flowers in packages of 3 for $2.50 or in packages of 2 for $1. "
        "How much money does he save by buying 18 flowers at the better price?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )
    pred, _ = generic_agent.solve(question)
    assert pred == "6"


def test_generic_symbolic_variant_handles_more_than_and_total_together() -> None:
    question = (
        "After transferring to a new school, Amy made 20 more friends than Lily. "
        "If Lily made 50 friends, how many friends do Lily and Amy have together?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(
            mode="direct",
            use_symbolic_solver=True,
            symbolic_solver_variant="generic",
            use_query_rewrite=False,
        ),
    )
    pred, _ = generic_agent.solve(question)
    assert pred == "120"


def test_generic_symbolic_variant_handles_daily_rate_minus_failures() -> None:
    question = "Ryan plants 2 flowers a day in his garden. After 15 days, how many flowers does he have if 5 did not grow?"
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(mode="direct", use_symbolic_solver=True, symbolic_solver_variant="generic", use_query_rewrite=False),
    )
    pred, _ = generic_agent.solve(question)
    assert pred == "25"


def test_generic_symbolic_variant_handles_more_than_times_base() -> None:
    question = (
        "In a candy machine, there are 22 more than four times the number of pink gumballs as there are blue gumballs. "
        "If there are 12 blue gumballs how many pink ones are there?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(mode="direct", use_symbolic_solver=True, symbolic_solver_variant="generic", use_query_rewrite=False),
    )
    pred, _ = generic_agent.solve(question)
    assert pred == "70"


def test_generic_symbolic_variant_handles_twice_target_with_over_and_drop() -> None:
    question = (
        "Henry is making cookies for a local baking competition. He wants to make twice as many as he did last year. "
        "When he finishes baking, he realizes he actually baked 15 more cookies than he meant to. "
        "He drops 5 of his cookies as he is putting them out to cool, and now has a total of 110 cookies. "
        "How many cookies did Henry bake last year?"
    )
    generic_agent = OrchestratedAgent(
        MockClient(seed=0),
        AgentConfig(mode="direct", use_symbolic_solver=True, symbolic_solver_variant="generic", use_query_rewrite=False),
    )
    pred, _ = generic_agent.solve(question)
    assert pred == "50"


