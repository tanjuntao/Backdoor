from typing import Dict

import torch


def optimize_input(
    images: Dict[str, torch.Tensor],
    models: dict,
    loss_fn,
    target,
    other,
    num_steps=50,
    agg="add",
    verbose=False,
    prob_gap=0.05,
):
    poison_image = images["passive"].unsqueeze(0)  # add batch dimension
    active_image = images["active"].unsqueeze(0)
    upper_bound, lower_bound = torch.max(poison_image), torch.min(poison_image)
    poison_image.requires_grad = True  # for optimization
    optimizer = torch.optim.Adam([poison_image], lr=0.1)
    # optimizer = torch.optim.SGD([poison_image], lr=0.01)

    for model in models.values():
        model.eval()

    for i in range(num_steps):
        optimizer.zero_grad()
        if agg == "add":
            active_repr = models["active_cut"](models["active_bottom"](active_image))
            passive_repr = models["passive_cut"](models["passive_bottom"](poison_image))
            top_input = active_repr + passive_repr
        elif agg == "concat":
            active_repr = models["active_bottom"](active_image)
            passive_repr = models["passive_bottom"](poison_image)
            top_input = torch.concat((active_repr, passive_repr), dim=1)
        else:
            raise ValueError(f"{agg} is not recognized agg() function.")
        logits = models["top"](top_input)
        _, predicted = logits.max(1)

        soft_label = torch.zeros_like(logits)
        soft_label[:, target] = 0.5 + prob_gap / 2
        soft_label[:, other] = 0.5 - prob_gap / 2
        total_loss = loss_fn(logits, soft_label)
        if verbose:
            print(
                f"step: {i:02d}, loss: {total_loss.item():.6f}, pred:"
                f" {predicted.item()}"
            )

        # loss_dict = loss_fn(logits)
        # total_loss, loss1, loss2 = loss_dict["loss"], loss_dict["loss1"], loss_dict["loss2"]
        # if verbose:
        #     print(
        #         f"step: {i:02d}, loss: {total_loss.item():.6f}, loss1: {loss1.item():.6f}, loss2: {loss2.item():.6f}, pred: {predicted.item()}"
        #     )
        if i == num_steps - 1:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().tolist()[0]
            # probs = [round(prob, 5) for prob in probs]
            for idx, prob in enumerate(probs):
                if idx == 0:
                    print("[", end="")
                print(f"{prob:.6f}", end=", ")
                if idx == len(probs) - 1:
                    print("]", end=" ")
            print(predicted.item())
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            poison_image.clamp_(lower_bound, upper_bound)

    poison_image.requires_grad = False
    return poison_image.squeeze()
