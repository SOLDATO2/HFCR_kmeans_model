import pandas as pd
from pickle import load
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd

def undummify(df):
    # Create an empty dictionary to store the undummified values
    undummified_dict = {}
    assigned_categories = set()  # Keep track of assigned categories
    
    for col in df.columns:
        # Split the column name by underscores
        parts = col.split("_")
        
        # Extract the original category name (all parts except the last one)
        category_name = "_".join(parts[:-1])
        
        # Get the value (1 or 0) from the dummy-encoded column
        value = df[col]
        
        # Check if the category name has already been assigned a value
        if category_name not in assigned_categories:
            if value[0] == 1:
                undummified_value = parts[-1]
                assigned_categories.add(category_name)  # Mark as assigned
                undummified_dict[category_name] = undummified_value
            else:
                undummified_value = None  # Handle cases where value is 0
                undummified_dict[category_name] = undummified_value
        
        # Store the undummified value in the dictionary

    
    # Create a new DataFrame from the dictionary
    undummified_df = pd.DataFrame(undummified_dict, index=[0])
    
    return undummified_df


# Carregar o modelo de normalização e colunas categóricas
HFCR_kmeans_model = load(open('HFCR_treinamento\\HFCR_clusters_2024.pkl', 'rb'))
modelo_normalizador = load(open('modelo_normalizador.pkl', 'rb'))
with open('HFCR_treinamento\\HFCR.csv', 'r') as file:
    # Ler a primeira linha do arquivo, que contém os nomes das colunas separados por vírgulas
    columns = file.readline().strip().split(',')

# Criar DataFrame vazio com as colunas obtidas
data_frame = pd.DataFrame(columns=columns)




app = Flask(__name__)


@app.route('/receber_informacoes', methods=['GET','POST'])
def receber_informacoes():

    if request.method == 'POST':
        dados = request.json
        rota = dados.get('rota')
        nova_instancia = dados.get('informacoes')







    #nova_instancia = [40, "no", 100, "no", 20, "no", 263358.03, 1.1, 100, "Male", "no", 4, "no"]


    # Iterando pela lista e modificando os elementos
    for i in range(len(nova_instancia)):
        if isinstance(nova_instancia[i], str) and "_" in nova_instancia[i]:
            nova_instancia[i] = nova_instancia[i].replace("_", "")

    print(nova_instancia)



    nova_instancia_dict = {
        'age': nova_instancia[0],
        'anaemia': nova_instancia[1],
        'creatinine_phosphokinase': nova_instancia[2],
        'diabetes': nova_instancia[3],
        'ejection_fraction': nova_instancia[4],
        'high_blood_pressure': nova_instancia[5],
        'platelets': nova_instancia[6],
        'serum_creatinine': nova_instancia[7],
        'serum_sodium': nova_instancia[8],
        'sex': nova_instancia[9],
        'smoking': nova_instancia[10],
        'time': nova_instancia[11],
        'DEATH_EVENT': nova_instancia[12]
    }

    # Convertendo para DataFrame
    nova_instancia_df = pd.DataFrame([nova_instancia_dict]) #Até aqui ele está inserindo os dados corretamente
    print(nova_instancia_df)

    # Separar dados numéricos e categóricos
    nova_instancia_numericos = nova_instancia_df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT'])
    nova_instancia_categoricos = nova_instancia_df[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']]

    #print(nova_instancia_df)
    #print(nova_instancia_numericos) ###################dados categoricos possuem 8 colunas##############################
    #print(nova_instancia_categoricos)


    ############################
    ############################
    # Normalizar dados numéricos
    nova_instancia_numericos_normalizados = modelo_normalizador.transform(nova_instancia_numericos) ##############continua tendo 8 elementos#############
    print(len(nova_instancia_numericos_normalizados[0]))
    # Aplicar one-hot encoding aos dados categóricos
    print(nova_instancia_categoricos)
    nova_instancia_categoricos_normalizados = pd.get_dummies(nova_instancia_categoricos, dtype=int) #Esta aplicando o get dummies corretamente
    print(nova_instancia_categoricos_normalizados)
    ############################
    ############################
    pd.set_option('display.max_columns', None)
    # Juntar dados normalizados e codificados
    # Gera um unico dataframe com todas as tabelas, primeiro existem no pd df as tabelas numericas e entao sao inseridas as tabelas categoricas
    nova_instancia_final_normalizada_df = pd.DataFrame(data=nova_instancia_numericos_normalizados, columns=nova_instancia_numericos.columns).join(nova_instancia_categoricos_normalizados)
    print(len(nova_instancia_final_normalizada_df.columns)) #Possui 17 colunas, ou seja, está juntando corretamente os numericos com os categoricos
    print(nova_instancia_final_normalizada_df)


    ################################################

    #O PROBLEMA ESTÁ AQUI, apos fazer o join em 'nova_instancia_final_normalizada_df', ainda falta as outras colunas dos dados
    #categricos que não foram incluidas apos o get dummies

    ################################################


    # Copiar o DataFrame original para manter a ordem das colunas
    # Alimentar as colunas existentes
    nova_instancia_final_normalizada_ORGANIZADA_df = data_frame.copy()
    nova_instancia_final_normalizada_ORGANIZADA_df[nova_instancia_final_normalizada_df.columns] = nova_instancia_final_normalizada_df
    # Preencher as colunas faltantes com zeros
    print(nova_instancia_final_normalizada_ORGANIZADA_df)
    nova_instancia_final_normalizada_ORGANIZADA_df = nova_instancia_final_normalizada_ORGANIZADA_df.fillna(0)
    print(nova_instancia_final_normalizada_ORGANIZADA_df)

    # Troca colunas nulas por 0



    #pega a primeira instancia da tabela
    instancia_normalizada_do_df_normalizado_organizado = nova_instancia_final_normalizada_ORGANIZADA_df.iloc[0]
    print(instancia_normalizada_do_df_normalizado_organizado)
    #pega os valores da instancia (esquerda para direita)
    valores_instancia_normalizada_do_df_normalizado_organizado = instancia_normalizada_do_df_normalizado_organizado.values
    print(valores_instancia_normalizada_do_df_normalizado_organizado)

    #armazena colunas para dar ao centroid futuramente
    ordem_colunas = nova_instancia_final_normalizada_ORGANIZADA_df.columns
    print(ordem_colunas)

    #print(len(valores_instancia_normalizada_do_df_normalizado_organizado)) #Possui 38 elementos, ou seja, 38 valores para 38 colunas
    #Contei manualmente as colunas normalizadas em "EOLBOEHPC.csv" e de fato existem 38 colunas após a normalização


    # nova instancia final em teoria deve estar com PRIMEIRO todos os valores numericos e então todos os valores categoricos

    indice_grupo_centroid_entrevistado = HFCR_kmeans_model.predict([valores_instancia_normalizada_do_df_normalizado_organizado])
    centroid_entrevistado = HFCR_kmeans_model.cluster_centers_[HFCR_kmeans_model.predict([valores_instancia_normalizada_do_df_normalizado_organizado])]

    print("Indice do grupo do novo entrevistado:", indice_grupo_centroid_entrevistado)
    print("Centroide do entrevistado: ", centroid_entrevistado)

    centroid = pd.DataFrame(HFCR_kmeans_model.cluster_centers_[HFCR_kmeans_model.predict([valores_instancia_normalizada_do_df_normalizado_organizado])])











    print(centroid) #Possui 38 colunas, bate com a quantidade no EOLBOEHPC.csv
    print("Quantidade de colunas em centroid:", len(centroid.columns))

    #1. Atribuir os rótulos do arquivo de treinamento ao centroid
    #2. Segmentar o centroid em numéricos e categóricos
    #3. Centroid_numericos = aplicar o inverse transform
    #4. Centroid_categoricos = aplicar o pd.from_dummies()


    # Atribuir os rótulos do arquivo de treinamento ao centroid
    centroid.columns = ordem_colunas
    print(centroid.columns)
    print("Quantidade de colunas em centroid:", len(centroid.columns))


    # Segmentar o centroid em numéricos e categóricos 
    print("##############################################")
    lista_colunas_categoricas_normalizadas_para_drop_centroid = []

    # Iterar sobre as colunas do DataFrame categorico normalizado
    for coluna in nova_instancia_categoricos_normalizados.columns:
        # Extrair o prefixo da coluna atual
        prefixo = coluna.split('_')[0]
        # Filtrar as colunas do DataFrame final normalizado que têm o mesmo prefixo
        colunas_prefixo = list(filter(lambda x: x.startswith(prefixo), nova_instancia_final_normalizada_ORGANIZADA_df.columns))
        # Adicionar as colunas encontradas à lista
        lista_colunas_categoricas_normalizadas_para_drop_centroid.extend(colunas_prefixo)
        
    print(lista_colunas_categoricas_normalizadas_para_drop_centroid)
    # Remover duplicatas da lista, se houver
    #colunas_categoricas_normalizadas_para_drop_centroid = list(set(lista_colunas_categoricas_normalizadas_para_drop_centroid))
    #print(len(colunas_categoricas_normalizadas_para_drop_centroid))



    centroid_colunas_numericas_normalizadas = centroid.drop(columns=lista_colunas_categoricas_normalizadas_para_drop_centroid)
    print(centroid_colunas_numericas_normalizadas)
    print("----------------------------------------------")

    centroid_colunas_categoricas_normalizadas = centroid[lista_colunas_categoricas_normalizadas_para_drop_centroid]
    print(centroid_colunas_categoricas_normalizadas)

    print("----------------------------------------------")
    print("##############################################")


    centroid_colunas_numericas_desnormalizadas = modelo_normalizador.inverse_transform(centroid_colunas_numericas_normalizadas)
    print(centroid_colunas_numericas_desnormalizadas)

    #4. Centroid_categoricos = aplicar o pd.from_dummies()

    print(centroid_colunas_categoricas_normalizadas)
    print("----------------------------------------------")
    #centroid_colunas_categoricas_normalizadas = centroid_colunas_categoricas_normalizadas.round()
    centroid_colunas_categoricas_normalizadas = centroid_colunas_categoricas_normalizadas.applymap(lambda x: 1 if x >= 0.45 else 0)
    centroid_colunas_categoricas_normalizadas = centroid_colunas_categoricas_normalizadas.astype(int)
    print(centroid_colunas_categoricas_normalizadas.iloc[0])
    #centroid_colunas_categoricas_desnormalizadas = pd.from_dummies(centroid_colunas_categoricas_normalizadas)

    centroid_colunas_categoricas_desnormalizadas = undummify(centroid_colunas_categoricas_normalizadas)
    print("----------------------------------------------")

    print(centroid_colunas_categoricas_desnormalizadas)



    nova_instancia_final_normalizada_df = pd.DataFrame(data=centroid_colunas_numericas_desnormalizadas.round(), columns=centroid_colunas_numericas_normalizadas.columns).join(centroid_colunas_categoricas_desnormalizadas)


    print("----------------------------------------------")


    print(nova_instancia_final_normalizada_df)
    
    
    
    dados = {
    "indice_grupo_centroid_entrevistado": indice_grupo_centroid_entrevistado.tolist(),
    "centroid_entrevistado": centroid_entrevistado.tolist(),
    "nova_instancia_final_normalizada_df": nova_instancia_final_normalizada_df.to_dict(orient='records')
    }

    # Enviar solicitação POST
    response = requests.post(rota, json=dados)

    # Verificar o status da resposta
    if response.status_code == 200:
        print("Informações enviadas com sucesso para a rota.")
    else:
        print("Ocorreu um erro ao enviar informações para a rota.")
    return "Operacao feita"




if __name__ == '__main__':
    app.run(host='localhost', port=5000)










