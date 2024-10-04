import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import openai
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

# Carregar as variáveis diretamente do Streamlit Secrets
aoai_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
aoai_key = st.secrets["AZURE_OPENAI_API_KEY"]
aoai_deployment_name = st.secrets["AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME"]
search_endpoint = st.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
search_key = st.secrets["AZURE_SEARCH_SERVICE_ADMIN_KEY"]
storage_account = st.secrets["AZURE_STORAGE_ACCOUNT"]
storage_container = st.secrets["AZURE_STORAGE_CONTAINER"]

# Instruções detalhadas para o assistente da Promon Engenharia
ROLE_INFORMATION = """
Instruções para o Assistente de IA da Promon Engenharia:

Contexto e Propósito:
Você é um assistente de inteligência artificial integrado à Promon Engenharia, com a função de auxiliar os usuários na consulta e extração de informações dos diversos documentos indexados da empresa. Esses documentos podem abranger diferentes projetos, diretrizes internas, normativos de recursos humanos e demais documentos relevantes para o funcionamento e operação da Promon.

Função Principal:
Seu papel é fornecer respostas precisas e relevantes com base nas informações disponíveis nos documentos internos da Promon. Esses documentos podem incluir detalhes técnicos de projetos, cronogramas, especificações, plantas, relatórios, diretrizes de recursos humanos, normativos internos, manuais de procedimentos e outros dados pertinentes à empresa.

Diretrizes para Respostas:

Consultas Baseadas em Documentos:

Todas as respostas devem ser baseadas exclusivamente nas informações contidas nos documentos internos aos quais você tem acesso. Ao receber uma consulta, identifique o índice ou pasta correspondente (projetos, diretrizes de RH, normativos, etc.) e forneça uma resposta clara e concisa baseada no conteúdo dos documentos disponíveis.
Abrangência dos Projetos e Documentos:

Você tem acesso a documentos relacionados a diversos projetos e departamentos da Promon, incluindo mas não se limitando a:
Projetos Técnicos: Forneça informações sobre detalhes técnicos, cronogramas, especificações e demais dados relevantes.
Normativos e Diretrizes Internas: Auxilie com informações sobre diretrizes de recursos humanos, normas internas, políticas da empresa e manuais de procedimentos.
Caso a consulta do usuário se refira a informações fora dos documentos disponíveis ou fora do escopo da sua atuação, responda com: "Não tenho acesso a essa informação".
Respostas Estruturadas:

Estruture suas respostas de forma clara, apresentando as informações de maneira organizada e fácil de entender. Utilize listas, tópicos numerados ou seções separadas quando necessário para facilitar a compreensão do usuário.
Ausência de Informações:

Se a informação solicitada pelo usuário não estiver disponível nos documentos que você pode consultar, ou não houver dados relacionados ao tema solicitado, responda diretamente com: "Não tenho acesso a essa informação".
Exemplos de Consultas:

Exemplo 1: Usuário: "Quais são as normas de segurança vigentes no projeto de Expansão da Área 6?"
Resposta: "As normas de segurança para o projeto de Expansão da Área 6 incluem o uso obrigatório de EPIs, controle de acesso a áreas restritas e inspeções regulares de equipamentos. Consulte o documento XYZ.pdf, seção 5.2, para mais detalhes."

Exemplo 2: Usuário: "Quais são as políticas de home office da Promon?"
Resposta: "As políticas de home office da Promon são definidas no documento 'Política de Trabalho Remoto 2024.pdf', que estabelece critérios como elegibilidade, frequência e ferramentas de apoio ao colaborador."

Exemplo 3: Usuário: "Qual é o prazo de entrega previsto para o Projeto XYZ?"
Resposta: "De acordo com o cronograma presente no documento 'Cronograma_Projeto_XYZ.pdf', a entrega está prevista para junho de 2025."

Considerações Finais:
Mantenha clareza, objetividade e relevância em todas as respostas. Garanta que o usuário receba as informações mais atualizadas e pertinentes, baseadas exclusivamente nos documentos disponíveis para consulta. Seu objetivo é facilitar o acesso a informações técnicas e administrativas, respeitando sempre os limites de acesso aos conteúdos indexados da Promon Engenharia.

"""

# --- USER AUTHENTICATION ---
# Nomes e usernames para autenticação
names = ["Peter Parker", "Rebecca Miller", "Michel Daros", "Gustavo Pelissaro", "Alex Sandoval", "Alexsandra Mendes", "Marco Lamim", "Guilherme Grandesi", "Henrique Riego", "Rogerio Ishikawa", "David Andrade", "Fabiana Garcia", "Gabriela Souza", "Andre Hiroshi", "Rafael Pereira", "Gisele Duarte", "Bruna Rufino", "Hellen Vitali", "Rosana Bretzel", "Maria Araújo"]
usernames = ["pparker", "rmiller", "mdaros", "gpelissaro", "asandoval", "amendes", "mlamim", "ggrandesi", "hriego", "rishikawa", "dandrade", "fgarcia", "gsouza", "ahiroshi", "rpereira", "gduarte", "brufino", "hvitali", "rbretzel", "maraujo"]

# Carregar as senhas hasheadas
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Configurar as credenciais de autenticação
credentials = {
    "usernames": {
        usernames[0]: {"name": names[0], "password": hashed_passwords[0]},
        usernames[1]: {"name": names[1], "password": hashed_passwords[1]},
        usernames[2]: {"name": names[2], "password": hashed_passwords[2]},
        usernames[3]: {"name": names[3], "password": hashed_passwords[3]},
        usernames[4]: {"name": names[4], "password": hashed_passwords[4]},
        usernames[5]: {"name": names[5], "password": hashed_passwords[5]},
        usernames[6]: {"name": names[6], "password": hashed_passwords[6]},
        usernames[7]: {"name": names[7], "password": hashed_passwords[7]},
        usernames[8]: {"name": names[8], "password": hashed_passwords[8]},
        usernames[9]: {"name": names[9], "password": hashed_passwords[9]},
        usernames[10]: {"name": names[10], "password": hashed_passwords[10]},
        usernames[11]: {"name": names[11], "password": hashed_passwords[11]},
        usernames[12]: {"name": names[12], "password": hashed_passwords[12]},
        usernames[13]: {"name": names[13], "password": hashed_passwords[13]},
        usernames[14]: {"name": names[14], "password": hashed_passwords[14]},
        usernames[15]: {"name": names[15], "password": hashed_passwords[15]},
        usernames[16]: {"name": names[16], "password": hashed_passwords[16]},
        usernames[17]: {"name": names[17], "password": hashed_passwords[17]},
        usernames[18]: {"name": names[18], "password": hashed_passwords[18]},
        usernames[19]: {"name": names[19], "password": hashed_passwords[19]}        
    }
}

# Criar o objeto de autenticação
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="promon_ai_chatbot",
    cookie_key="some_cookie_key",
    cookie_expiry_days=30
)

# Função para carregar índices do Azure AI Search
def get_available_indexes(search_endpoint, search_key):
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_key))
    indexes = index_client.list_indexes()  # Lista todos os índices disponíveis
    return [index.name for index in indexes]

# Mapeamento entre os nomes reais dos índices e os nomes amigáveis
index_name_mapping = {
    "epotl-dp": "E.POTL001 - Projeto GLP/C5+",
    "vopak-dp": "E.VPAK001 - VOPAK",
    "recursos-humanos": "Relações Humanas"
}

# Função para obter o nome amigável a partir do nome real do índice
def get_friendly_index_name(real_index_name):
    return index_name_mapping.get(real_index_name, real_index_name)

# Função para criar o chat com dados do Azure AI Search
def create_chat_with_data_completion(aoai_deployment_name, messages, aoai_endpoint, aoai_key, search_endpoint, search_key, selected_index):
    client = openai.AzureOpenAI(
        api_key=aoai_key,
        api_version="2024-06-01",
        azure_endpoint=aoai_endpoint
    )
    return client.chat.completions.create(
        model=aoai_deployment_name,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        stream=True,
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "index_name": selected_index,  # Usar o índice selecionado pelo usuário
                        "semantic_configuration": "default",
                        "query_type": "semantic",
                        "fields_mapping": {},
                        "in_scope": True,
                        "role_information": ROLE_INFORMATION,
                        "strictness": 3,
                        "top_n_documents": 5,
                        "authentication": {
                            "type": "api_key",
                            "key": search_key
                        }
                    }
                }
            ]
        }
    )

# Função para lidar com a entrada do chat e gerar resposta
def handle_chat_prompt(prompt, aoai_deployment_name, aoai_endpoint, aoai_key, search_endpoint, search_key, selected_index):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Exibir a resposta do chatbot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        documents_used = []

        for response in create_chat_with_data_completion(aoai_deployment_name, st.session_state.messages, aoai_endpoint, aoai_key, search_endpoint, search_key, selected_index):
            if response.choices:
                full_response += (response.choices[0].delta.content or "")
                if hasattr(response.choices[0], 'data') and "documents" in response.choices[0].data:
                    documents_used = response.choices[0].data["documents"]
                message_placeholder.markdown(full_response + "▌")

        # Gerar links clicáveis para os documentos utilizados, caso existam
        if documents_used:
            full_response += "\n\n**Documentos Referenciados:**\n"
            for i, doc in enumerate(documents_used):
                doc_url = f"https://{storage_account}.blob.core.windows.net/{storage_container}/{doc['document_id']}"
                full_response += f"[{i+1}]({doc_url})\n"

        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Função principal do Streamlit

def main():
    st.title("MakrAI - Assistente Virtual Promon")

    # Inicializar o histórico de mensagens se ainda não foi feito
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Autenticação na barra lateral
    try:
        name, authentication_status, username = authenticator.login("Login", "sidebar")
    except Exception as e:
        st.error(f"Erro durante o login: {str(e)}")
        return

    # Verificar o status da autenticação
    if authentication_status is False:
        st.error("Nome de usuário ou senha incorretos")
    elif authentication_status is None:
        st.warning("Por favor, insira o nome de usuário e a senha")
    else:
        # --- SE O USUÁRIO ESTIVER AUTENTICADO ---
        st.success(f"Bem-vindo, {name}, ao MakrAI!")

        # Carregar índices disponíveis do Azure AI Search
        available_indexes = get_available_indexes(search_endpoint, search_key)

        if not available_indexes:
            st.error("Nenhum índice disponível no Azure Search.")
            return

        # Criar uma lista de nomes amigáveis a serem exibidos no dropdown
        friendly_indexes = [get_friendly_index_name(index) for index in available_indexes]

        # Dropdown para selecionar o índice
        selected_friendly_index = st.sidebar.selectbox("Selecione o índice do Azure AI Search", options=friendly_indexes)

        # Encontrar o nome real do índice selecionado com base no nome amigável
        selected_index = next((key for key, value in index_name_mapping.items() if value == selected_friendly_index), selected_friendly_index)

        # Exibir o histórico de mensagens
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Caixa de entrada do chat
        if prompt := st.chat_input("Digite sua pergunta:"):
            handle_chat_prompt(prompt, aoai_deployment_name, aoai_endpoint, aoai_key, search_endpoint, search_key, selected_index)

        # Adicionar disclaimer no rodapé
        st.sidebar.markdown("""
        **Disclaimer**:
        O "MakrAI" tem como único objetivo disponibilizar dados que sirvam como um meio de orientação e apoio; não constitui, porém, uma recomendação vinculante pois não representam uma análise personalizada para um Cliente e/ou Projeto específico, e, portanto, não devem ser utilizados como única fonte de informação na tomada de decisões pelos profissionais Promon.
        """)

        # Botão de logout
        authenticator.logout("Logout", "sidebar")
