import time
from flask import Flask, request, jsonify
import requests
import threading

app = Flask(__name__)

def enviar_dados():
    minha_rota = f'http://localhost:{5001}/receber_informacoes_do_clustering'  # Obtenha a porta atual do validador.py
    informacoes = []
    escolha = input("Gostaria de enviar um dado pre-definidos para enviar ao clustering? (yes/no): ")
    if escolha == "no":
        info = int(input("Qual é a idade do examinado?: "))
        informacoes.append(info)
        info =  input("O examinado possui anemia?(yes/no): ")
        informacoes.append(info)
        info = int(input("Qual é o nivel de creatina fosfoquinase do examinado? (ex: 340): "))
        informacoes.append(info)
        info = input("O examinado possui diabetes?(yes/no): ")
        informacoes.append(info)
        info = int(input("Qual é o nivel de fração de ejeção do examinado? (ex: 30): "))
        informacoes.append(info)
        info =  input("O examinado possui pressão alta?(yes/no): ")
        informacoes.append(info)
        info = float(input("Qual é o nivel de plaquetas do examinado? (ex: 271000.00): "))
        informacoes.append(info)
        info = float(input("Qual é o nivel de resíduos químicos no sangue do examinado? (ex: 0.5): "))
        informacoes.append(info)
        info = float(input("Qual é o nivel de sódio no sangue do examinado? (ex: 130): "))
        informacoes.append(info)
        info = input("Qual é o sexo do examinado? (Female/Male): ")
        informacoes.append(info)
        info =  input("O examinado fuma?(yes/no): ")
        informacoes.append(info)
        info = int(input("Quantos dias se passaram apos o ataque de coracao? (ex: 3): "))
        informacoes.append(info)
        info = input("O examinado morreu apos o ataque?(yes/no): ")
    elif escolha == "yes":
        informacoes = [40, "no", 100, "no", 20, "no", 263358.03, 1.1, 100, "Male", "no", 4, "no"]
    
    
    print("Enviando dados...")
    
    try:
        response = requests.post('http://localhost:5000/receber_informacoes', json={'rota': minha_rota, 'informacoes': informacoes})
        if response.status_code == 200:
            print("Dados enviados com sucesso para o clustering.")
        else:
            print("Falha ao enviar dados para o clustering. Tentando novamente...")
    except requests.exceptions.RequestException as e:
        print("Erro de conexão:", e)
    time.sleep(5)


@app.route('/receber_informacoes_do_clustering', methods=['POST'])
def receber_informacoes_do_clustering():
    
    if request.method == 'POST':
        dados = request.json
        indice_grupo_centroid_entrevistado = dados.get('indice_grupo_centroid_entrevistado')
        centroid_entrevistado = dados.get('centroid_entrevistado')
        nova_instancia_final_normalizada_df = dados.get('nova_instancia_final_normalizada_df')
    
    print("Indicie do grupo do centroid do examinado: ", indice_grupo_centroid_entrevistado)
    print("Centroid do examinado: ", centroid_entrevistado)
    print("Instancia retornada do centroid: ", nova_instancia_final_normalizada_df[0])
    return "Operacao feita com sucesso"
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    thread_informacoes = threading.Thread(target=enviar_dados)
    thread_informacoes.start()
    
    app.run(host='localhost', port=5001)