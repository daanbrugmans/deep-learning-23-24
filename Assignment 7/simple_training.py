import dl_assignment_7_common as common
import lunar_phase

from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from d2l.torch import try_gpu


def main():
    arch_name = "lenet"
    dataset_name = "fashionmnist"
    run_name = "1"
    lunar_phase_name = lunar_phase.get_lunar_phase_name()
    model_name = f"{arch_name}-{dataset_name}-{run_name}-{lunar_phase_name}"
    
    device_name = try_gpu()
    net = common.create_network(arch_name)
    data_loaders = common.get_dataset(dataset_name, used_data=0.01)
    
    common.train(
        net,
        data_loaders,
        optimizer=Adam,
        learning_rate=0.0012,
        loss_fn=CrossEntropyLoss(),
        epochs=10,
        device=device_name,
        model_file_name=model_name,
        eval_every_n_epochs=2
    )


if __name__ == '__main__':
    main()
