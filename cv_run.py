import cv2
import numpy as np
import torch

from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im

models = [
    "style_monet_pretrained",
    "style_vangogh_pretrained",
    "style_cezanne_pretrained",
    "style_ukiyoe_pretrained",
]


def on_trackbar(val):
    print("Switching to", models[int(val)])


def reset_model(opt, model_idx):
    opt.name = models[model_idx]

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    return model


def main(opt, name="CycleGAN", model_idx=0):
    model = reset_model(opt, model_idx)

    cv2.namedWindow(name)
    cv2.createTrackbar(
        "Model Selector",
        name,
        model_idx,
        3,
        on_trackbar,
    )
    cap = cv2.VideoCapture(0)

    data = {"A": None, "A_paths": None}
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(np.array([frame]), (0, 3, 1, 2))

        data["A"] = torch.from_numpy(frame).float()
        model.set_input(data)
        model.test()

        image = model.get_current_visuals()["fake"]
        image = tensor2im(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.putText(
            image,
            models[model_idx],
            (10, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow(name, image)

        key = cv2.waitKey(1)
        if key == 27:
            break

        val = cv2.getTrackbarPos("Model Selector", name)
        if val != model_idx:
            model_idx = val
            model = reset_model(opt, model_idx)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    main(opt)
