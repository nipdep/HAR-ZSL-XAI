import torch


def get_config(file_loc):
    file = torch.load(file_loc)
    return file["model_state_dict"], file["model_config"], file["config"]


if __name__ == "__main__":
    get_config("D:\\FYP\\HAR-ZSL-XAI\\model_saves\\bidiretional_lstm_hrnet_nturgb\\epoch10_emb1024_xy.pt")
