import wandb
import statistics

def log_metrics(step, args, task_list, train_loss_list_of_list, test_loss_list_of_list, test_acc_list_of_list, mean_acc_list, last_acc_list, loss_list, last_point_loss_list, plateau_logged, complete_logged, init_distance):
    wandb.log({"init_distance": init_distance, "total_loss": sum(loss_list)}, step=step)

    for j, task in enumerate(task_list):
        task_name = task['exp_name']
        train_loss_list_of_list[task_name].append(loss_list[j])

        if task['data'] == 'gaussian':
            test_loss_list_of_list[task_name].append(last_point_loss_list[j])
        else:
            test_acc_list_of_list[task_name].append(last_acc_list[j])

        tracking_loss_list = list(train_loss_list_of_list[task_name])

        # Plateau Check
        if (not plateau_logged[task_name]) and statistics.mean(tracking_loss_list) < 0.8:
            plateau_logged[task_name] = True
            wandb.log({f"plateau_step_{task_name}": step}, step=step)

        # Completion Check
        if task['data'] == 'gaussian':
            threshold = 0.2 if task_name == 'quadratic_regression' else 0.1
            reached = statistics.mean(test_loss_list_of_list[task_name]) < threshold
        else:
            reached = statistics.mean(test_acc_list_of_list[task_name]) > 0.95

        if (not complete_logged[task_name]) and reached:
            complete_logged[task_name] = True
            wandb.log({f"complete_step_{task_name}": step}, step=step)

        # Log task-specific metrics
        wandb.log({
            f"mean_acc": sum(mean_acc_list) / len(mean_acc_list),
            f"mean_acc-{task_name}": mean_acc_list[j],
            f"last_acc-{task_name}": last_acc_list[j],
            f"overall_loss-{task_name}": loss_list[j],
            f"last_point_loss-{task_name}": last_point_loss_list[j],
        }, step=step)
