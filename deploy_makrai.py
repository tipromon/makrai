import os
import streamlit as st
import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import urllib.parse  # Para codificar URLs corretamente
import logging

# Configuração do logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        documents_used = []

        # Realizar a busca no Azure AI Search
        search_client = SearchClient(search_endpoint, selected_index, credential=AzureKeyCredential(search_key))
        logger.debug(f"Realizando busca no índice: {selected_index}")
        logger.debug(f"Prompt de busca: {prompt}")
        
        try:
            results = search_client.search(prompt, top=5, include_total_count=True)
            logger.debug(f"Total de resultados encontrados: {results.get_count()}")
        except Exception as e:
            logger.error(f"Erro ao realizar a busca: {str(e)}")
            results = []

        # Processar os documentos retornados da busca
        for result in results:
            doc_name = result.get('sourcefile', 'Documento sem nome')
            logger.debug(f"Documento encontrado: {doc_name}")
            logger.debug(f"Campos do documento: {result.keys()}")
            documents_used.append({
                'content': result.get('content', ''),
                'sourcefile': doc_name,
                'metadata': {k: v for k, v in result.items() if k not in ['content', 'sourcefile']}
            })

        # Processar a resposta do Azure OpenAI com integração ao Azure AI Search
        for response in create_chat_with_data_completion(aoai_deployment_name, st.session_state.messages, aoai_endpoint, aoai_key, search_endpoint, search_key, selected_index):
            if response.choices:
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")

        # Adicionar referências clicáveis ao final da resposta
        if documents_used:
            full_response += "\n\nReferências:\n"
            for i, doc in enumerate(documents_used):
                doc_name = os.path.basename(doc['sourcefile'])
                
                # Usar o container selecionado pelo usuário
                selected_container = selected_index  # Assumindo que o nome do índice corresponde ao container
                
                doc_url = f"https://{storage_account}.blob.core.windows.net/{selected_container}/{urllib.parse.quote(doc_name)}"
                logger.debug(f"URL do documento gerada: {doc_url}")
                logger.debug(f"Metadados do documento: {doc['metadata']}")

                # Criar link clicável
                full_response += f"{i+1}. [{doc_name}]({doc_url})\n"

        # Atualiza a resposta final no placeholder
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Log final para depuração
    logger.debug(f"Resposta completa gerada com {len(documents_used)} documentos referenciados.")

# Função principal do Streamlit
def main():
    st.title("MakrAI - Assistente Virtual Promon")

    # Carregar índices disponíveis do Azure AI Search
    available_indexes = get_available_indexes(search_endpoint, search_key)

    # Criar uma lista de nomes amigáveis a serem exibidos no dropdown
    friendly_indexes = [get_friendly_index_name(index) for index in available_indexes]

    # Dropdown para selecionar o índice
    selected_friendly_index = st.sidebar.selectbox("Selecione o índice do Azure AI Search", options=friendly_indexes)

    # Encontrar o nome real do índice selecionado com base no nome amigável
    selected_index = next((key for key, value in index_name_mapping.items() if value == selected_friendly_index), selected_friendly_index)

    # Inicializar o histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir o histórico de mensagens
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

if __name__ == "__main__":
    main()

