import src.continuous_task as continuous_task
import src.bool_task as bool_task

def build_task_train_list(task_list, n_dims, curriculum):
    task_train_list = []
    for task in task_list:
        if 'bool' not in task['data']:
            task_sampler = continuous_task.get_task_sampler(n_dims, n_points=curriculum.n_points, **task)
        else:
            task_sampler = bool_task.get_task_sampler(n_dims, n_points=curriculum.n_points, **task)
        task_train_list.append(task_sampler())
    return task_train_list
