import os
import pandas as pd

# Criar tabela

# os.makedirs("./tabelas", exist_ok=True)

# colunas = ["Age","Height","Weigth","Gender","Real distance", "Calculated distance","Erro"]
# df = pd.DataFrame(columns=colunas)

# df.to_excel("./tabelas/sit_and_reach_2_utentes.xlsx", index=False, engine="openpyxl")

# print("Arquivo Excel criado com sucesso!")




                        # caminho_arquivo = "./tabelas/dados.xlsx"
                        # df = pd.read_excel(caminho_arquivo, engine="openpyxl")
                        
                        # real = input("Qual a distância real: ")
                        # erro = np.abs(float(real) - float(final_distance))
                        # nova_linha = {
                        #             "Distância real": real,
                        #             "Distância calculada": final_distance,
                        #             "Erro": erro
                        #         }
                        # df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
                        # df.to_excel(caminho_arquivo, index=False, engine="openpyxl")


# Ler da tabela

# arquivo = "./tabelas/back_scratch_utentes.xlsx"
# df = pd.read_excel(arquivo)

# coluna = df["Erro"].values
# tamanho = df["Erro"].size + 1
# soma = 0

# for col in coluna:
#     soma += col 


# print(f"O erro aproximado é de {soma/tamanho:.3f}")

arquivo1 = "./tabelas_utentes/back_scratch_utentes.xlsx"
arquivo2 = "./tabelas_utentes/sit_and_reach_2_utentes.xlsx"

df = pd.read_excel(arquivo1)

coluna = df["Erro"].values
tamanho = df["Erro"].size + 1
soma = 0

for col in coluna:
    soma += col 

erro1 = soma/tamanho

print(f"O erro do Back-Scratch aproximado é de {erro1:.3f}")

df = pd.read_excel(arquivo2)

coluna = df["Erro"].values
tamanho = df["Erro"].size + 1
soma = 0

for col in coluna:
    soma += col   

erro2 = soma/tamanho

print(f"O erro do Sit and Reach aproximado é de {erro2:.3f}")

print(f"O erro dos dois programas é aproximado de {(erro1 + erro2) / 2:.3f}")