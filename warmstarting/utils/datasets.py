def select_dataset_id(dataset: str):
    if dataset == "IRIS":
        return 61
    if dataset == "CreditCardFraudDetection":
        return 42175
    else:
        return dataset