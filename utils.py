import pfrl

import wandb


class EvalWandbHook(pfrl.experiments.EvaluationHook):
    support_train_agent = True

    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
        print("Original Hook")
        wandb.log({"step": step, "eval/mean": eval_stats["mean"]})
