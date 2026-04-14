import pandas as pd

df = pd.read_excel('./tabelas_utentes/back_scratch_utentes.xlsx') 

df.columns = df.columns.str.strip()

agrupado = df.groupby(['Gender', 'Side'])['Erro'].agg(['mean', 'std', 'min', 'max'])

agrupado.columns = ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo']

print(agrupado)

agrupado.to_excel('./analises/estatisticas_erro_por_genero_e_lado_backscratch.xlsx')

print(agrupado.to_latex())
