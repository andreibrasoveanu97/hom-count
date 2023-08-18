import random
import time 
import os

import wandb
import torch
import numpy as np
from datetime import datetime, date

from Exp.parser import parse_args
import Misc.config as config
from Misc.utils import list_of_dictionary_to_dictionary_of_lists
from Exp.utils import load_dataset, get_model, get_optimizer_scheduler, get_loss
from Exp.training_loop_functions import train, eval, step_scheduler


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    now = datetime.now()
    model_path = os.path.join(config.RESULTS_PATH, "Models", f"{args.dataset}_{date.today()}_{now.strftime('%H:%M:%S')}")

    print(args)
    train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_vertex_features = train_loader.dataset.num_classes, train_loader.dataset.num_node_features
    print(f"Number of features: {num_vertex_features}")

    if "graph_features" in train_loader.dataset[0]:
        graph_features = train_loader.dataset[0].graph_features.shape[1]
    else:
        graph_features = 0

    if args.dataset.lower() == "zinc" or "ogb" in args.dataset.lower():
        num_classes = 1
    else:
        print(f"Classes: {train_loader.dataset.num_classes}")

    try:
        num_tasks = train_loader.dataset.num_tasks
        
    except:
        num_tasks = 1
        
    print(f"Tasks: {num_tasks}")

    model = get_model(args, num_classes, num_vertex_features, num_tasks)
    device = args.device
    use_tracking = args.use_tracking
    

    if args.use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config = args,
            project = "HomCount23")
        
    results = []

    

    finetune = False
    time_start = time.time()

    if graph_features > 0:
        nr_features = [0, graph_features] 
    else:
        nr_features = [0]

    for graph_feat in nr_features:
        
        if finetune:
            # Load model
            model.set_mlp(0)
            model.load_state_dict(torch.load(model_path))

            if args.do_freeze_gnn:
                model.freeze_gnn(True)

            model.set_mlp(graph_feat, copy_emb_weights = True)
            print(f"\n\n\nRetraining. Frozen: {args.do_freeze_gnn}, Graph features: {graph_feat}")
        
        optimizer, scheduler = get_optimizer_scheduler(model, args, finetune = finetune)
        loss_dict = get_loss(args)

        loss_fct = loss_dict["loss"]
        eval_name = loss_dict["metric"]
        if eval_name in ["mae", "rmse (ogb)"]:
            mode = "min"
        else:
            mode = "max"

        metric_method = loss_dict["metric_method"] 

        model.to(device)
        nr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {nr_parameters}")
        if nr_parameters > args.max_params:
            raise ValueError("Number of model parameters is larger than the allowed maximum")

        train_results, val_results, test_results = [], [], []
        best_val = float('inf') if mode == "min" else float('-inf')
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}")
            train_result = train(model, device, train_loader, optimizer, loss_fct, eval_name, use_tracking, metric_method=metric_method)
            val_result = eval(model, device, val_loader, loss_fct, eval_name, metric_method=metric_method)
            test_result = eval(model, device, test_loader, loss_fct, eval_name, metric_method=metric_method)

            train_results.append(train_result)
            val_results.append(val_result)
            test_results.append(test_result)

            if (mode == "min" and val_result[eval_name] < best_val) or (mode == "max" and val_result[eval_name] > best_val):
                best_val = val_result[eval_name]

                if not finetune:
                    print("Storing", model_path)
                    torch.save(model.state_dict(), model_path)

            # print(f"\tTRAIN \tLoss: {train_result['total_loss']:10.4f}\t{eval_name}: {train_result[eval_name]:10.4f}")
            print(f"\tTRAIN \tLoss: {train_result['total_loss']:10.4f}")

            print(f"\tVAL \tLoss: {val_result['total_loss']:10.4f}\t{eval_name}: {val_result[eval_name]:10.4f}")
            print(f"\tTEST \tLoss: {test_result['total_loss']:10.4f}\t{eval_name}: {test_result[eval_name]:10.4f}")

            if args.use_tracking:
                wandb.log({
                    "Epoch": epoch,
                    "Train/Loss": train_result["total_loss"],
                    # f"Train/{eval_name}": train_result[eval_name],
                    "Val/Loss": val_result["total_loss"],
                    f"Val/{eval_name}": val_result[eval_name],
                    "Test/Loss": test_result["total_loss"],
                    f"Test/{eval_name}": test_result[eval_name],
                    "LearningRate": optimizer.param_groups[0]['lr'],
                    "GraphFeatures": graph_feat
                    })

            step_scheduler(scheduler, args, val_result["total_loss"])

            # Exit conditions
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                    print("\nLR REACHED MINIMUM: Stopping")
                    break
                
        finetune = True

        # Final result
        train_results = list_of_dictionary_to_dictionary_of_lists(train_results)
        val_results = list_of_dictionary_to_dictionary_of_lists(val_results)
        test_result = list_of_dictionary_to_dictionary_of_lists(test_results)

        
        if mode == "min":
            best_val_epoch = np.argmin(val_results[eval_name])
        else:
            best_val_epoch = np.argmax(val_results[eval_name])

        val_results["best_epoch"] = best_val_epoch


        loss_train, loss_val, loss_test = train_results['total_loss'][best_val_epoch], val_results['total_loss'][best_val_epoch], test_result['total_loss'][best_val_epoch]
        result_val, result_test = val_results[eval_name][best_val_epoch], test_result[eval_name][best_val_epoch]

        wandb.log({
            f"Final/Val(graph_feat)/{eval_name}": result_val,
            f"Final/Test(graph_feat)/{eval_name}": result_test,
            f"graph_features": graph_feat})


        print("\n\nFINAL RESULT")
        runtime = (time.time()-time_start)/3600
        print(f"\tRuntime: {runtime:.2f}h")
        print(f"\tBest epoch {best_val_epoch + 1} / {args.epochs}")
        print(f"\tTRAIN \tLoss: {loss_train:10.4f}")
        print(f"\tVAL \tLoss: {loss_val:10.4f}\t{eval_name}: {result_val:10.4f}")
        print(f"\tTEST \tLoss: {loss_test:10.4f}\t{eval_name}: {result_test:10.4f}")

        results.append({
            "graph_features": graph_feat,
            "best_epoch": int(best_val_epoch + 1),
            "epochs": int(epoch),
            "final_val_result": result_val,
            "final_test_result": result_test,
            "final_train_loss": loss_train,
            "final_val_loss": loss_val,
            "final_test_loss": loss_test,
            "parameters": nr_parameters,
            "loss_train": train_results['total_loss'], 
            "loss_val": val_results['total_loss'],
            "loss_test": test_result['total_loss'],
            "best_result_val": val_results[eval_name][best_val_epoch],
            "result_val": val_results[eval_name],
            "result_test": test_result[eval_name],
            "runtime_hours":  runtime,
        })

    print("\nGraph features\tVal Result\tTest Result")
    for result in results:
        print(f"{result['graph_features']}: {result['final_val_result']}\t{result['final_test_result']}")

    val_metrics = list(map(lambda r: r["final_val_result"], results))
    test_metrics = list(map(lambda r: r["final_test_result"], results))
    graph_features = list(map(lambda r: r["graph_features"], results))


    # Do not use the 0 graph_feat count
    if args.graph_feat != "":
        best_val_idx = 1 + (np.argmin(val_metrics[1:]) if mode == "min" else np.argmax(val_metrics[1:]))
    else:
        best_val_idx = np.argmin(val_metrics) if mode == "min" else np.argmax(val_metrics)
    
    best_val_metric = val_metrics[best_val_idx]
    best_test_metric = test_metrics[best_val_idx]
    best_graph_feat = graph_features[best_val_idx]

    print(f"\nBest validation: {best_val_metric}\nBest test: {best_test_metric}\nGraph feat: {best_graph_feat}\n")

    if args.use_tracking:
        print("logging: ", end="")
        wandb.log({
            f"Best/Val-{eval_name}": best_val_metric,
            f"Best/Test-{eval_name}": best_test_metric,
            f"Best/Graph_feat": best_graph_feat})
        wandb.finish()
        print("Done.")

    # Delete model
    os.remove(model_path)

    return {"val": best_val_metric, 
        "test": best_test_metric,
        "graph_feat": best_graph_feat, 
        "runtime_hours":  (time.time()-time_start)/3600, 
        "parameters": nr_parameters,  
        "mode": mode, 
        "val0": results[0]["final_val_result"],
        "test0": results[0]["final_test_result"],
        "results": results,}
                

def run(passed_args = None):
    args = parse_args(passed_args)
    return main(args)

if __name__ == "__main__":
    run()