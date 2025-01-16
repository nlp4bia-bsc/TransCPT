import cosmos # type: ignore

if __name__ == "__main__":
    cosmos.initialization()

    result = cosmos.run(
        module_path="trans_cpt.training",
        function_name="training_pipeline",
        queue="acc_debug",
        user="bsc14",
        args=[{
            "data_path": "/gpfs/projects/bsc14/abecerr1/datasets/DT4H___wikipedia_cardiology_es/default/0.0.0/b20f70bf02ea8c0f5e0181e333b7b9ab3c610c4f",
        }],
        requirements=[
            "datasets",
            "transformers",
            "torch",
            "accelerate",
            "tqdm",
            "tensorboard"
        ],
        modules=[
            "cuda/12.6"
        ],
        partition="debug",
        nodes=1,
        cpus=80,
        gpus=4,
        venv_path="/gpfs/projects/bsc14/environments/trans_cpt",
        watch=True,
    )

