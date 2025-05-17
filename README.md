# ifood-case

Este repo tem o intuito de treinar um modelode recomendação de ofertas.

Para rodá-lo:

1 - Crie um venv com o arquivo `conda-env.yml`:
```
conda env create -f conda-env.yml
```

Após a instalação dos pacotes necessários, ative-o:
```
conda activate ifood-env
```

Apartir disso, basta rodar o arquivo ```python main.py``` que ele irá:
- Carregar os dados salvos em `data/raw`, processá-los e salvar em `data/processed`
- Construir um dataset unificado para a modelagem (`data/processed/modelling_dataset`)
- Treinar um modelo com os hiperparametros já otimizados via notebook e salvá-lo em `models_artefact`

