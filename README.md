# Tribrain
TriBrain is a three-agent (“Executive”, “Bayesian”, “Metacognitive”) world-model controller that evaluates episodes with real critics, stores them in SQLite episodic memory with atomic/idempotent migrations, and improves via replay, preference-based reward learning, and cost-bounded inference loops (incl. optional WoW subprocess integration).
