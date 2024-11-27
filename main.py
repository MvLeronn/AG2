import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def classificar_dados():
    print("Insira os valores para cada atributo:")
    region = int(input("Region (0 = Lisbon, 1 = Oporto, 2 = Other): "))
    fresh = float(input("Fresh (gasto anual em u.m.): "))
    milk = float(input("Milk (gasto anual em u.m.): "))
    grocery = float(input("Grocery (gasto anual em u.m.): "))
    frozen = float(input("Frozen (gasto anual em u.m.): "))
    detergents_paper = float(input("Detergents Paper (gasto anual em u.m.): "))
    delicatessen = float(input("Delicatessen (gasto anual em u.m.): "))

    # Cria um DataFrame com os valores de entrada
    input_data = pd.DataFrame([[region, fresh, milk, grocery, frozen, detergents_paper, delicatessen]], 
                              columns=['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents Paper', 'Delicatessen'])

    # Realiza a previsão com o modelo
    resultado = model.predict(input_data)
    
    # Interpreta o resultado
    canal = "HoReCa" if resultado[0] == 0 else "Retail"
    print(f"O canal de vendas previsto é: {canal}")

# Leitura do arquivo CSV
data = pd.read_csv('wholesale.csv')

# Conversão de acordo com o mapeamento
data['Channel'] = data['Channel'].replace({'HoReCa': 0, 'Retail': 1})
data['Region'] = data['Region'].replace({'Lisbon': 0, 'Oporto': 1, 'Other': 2})

# Reordena as colunas conforme a ordem desejada
data = data.reindex(columns=['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents Paper', 'Delicatessen', 'Channel'])

# Separação do target
X = data.drop('Channel', axis=1)  # remoção da variável-alvo do conjunto de features
y = data['Channel']               # separado como target,o valor que queremos prever

# Divisão em 80% para treinamento e 20% para testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando o classificador
model = DecisionTreeClassifier()

# Treinando o modelo
model.fit(X_train, y_train)

# Realizando previsões
y_pred = model.predict(X_test)

# Exibindo as métricas de avaliação
print(classification_report(y_test, y_pred))

# Chama a função para permitir ao usuário inserir dados arbitrários e classifica-los pelo modelo
classificar_dados()

