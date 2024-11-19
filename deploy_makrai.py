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

# Mapeamento de índicesw
index_mapping = {
    "E.VPAK001 - VOPAK": "vector-vpak",
    "E.POTL001 - Projeto GLP/C5+": "vector-epotl",
    "Relações Humanas": "vector-rh",
    "Inteligência de Mercado": "vector-bi"
}

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
    "vector-dp": "E.POTL001 - Projeto GLP/C5+",
    "vector-vpak": "E.VPAK001 - VOPAK",
    "vector-rh": "Relações Humanas",
    "vector-bi": "Inteligência de Mercado"
}

# Função para obter o nome amigável a partir do nome real do índice
def get_friendly_index_name(real_index_name):
    return index_name_mapping.get(real_index_name, real_index_name)

# Função para criar o chat com dados do Azure AI Search
def create_chat_with_data_completion(aoai_deployment_name, messages, aoai_endpoint, aoai_key, search_endpoint, search_key, selected_index):
    client = openai.AzureOpenAI(
        api_key=aoai_key,
        api_version="2024-02-15-preview",
        azure_endpoint=aoai_endpoint
    )
    try:
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
                            "index_name": selected_index,
                            "query_type": "simple",  # Alterado para busca simples
                            "fields_mapping": {
                                "content_fields": ["chunk"],
                                "title_field": "title",
                                "url_field": "parent_id"
                            },
                            "in_scope": True,
                            "role_information": ROLE_INFORMATION,
                            "top_n_documents": 5,
                            "filter": None,
                            "query_type_options": {
                                "speller": "lexicon",
                                "enable_fuzziness": True,
                                "fuzzy_min_similarity": 0.6
                            },
                            "authentication": {
                                "type": "api_key",
                                "key": search_key
                            }
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Erro ao chamar a API do OpenAI: {str(e)}")
        st.error("Ocorreu um erro ao processar sua solicita��ão. Por favor, tente novamente mais tarde.")
        raise
        
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
            # Atualizar os campos na busca para corresponder ao índice
            results = search_client.search(
                search_text=prompt,
                select=["chunk_id", "parent_id", "chunk", "title"],
                top=5,
                include_total_count=True
            )
            logger.debug(f"Total de resultados encontrados: {results.get_count()}")
        except Exception as e:
            logger.error(f"Erro ao realizar a busca: {str(e)}")
            results = []

        # Processar os documentos retornados da busca
        for result in results:
            # Usar os campos corretos do índice
            doc_name = result.get('title', '') or result.get('parent_id', 'Documento sem nome')
            content = result.get('chunk', '')
            
            logger.debug(f"Documento encontrado - título: {doc_name}")
            logger.debug(f"Conteúdo: {content[:100]}...")  # Log dos primeiros 100 caracteres
            
            documents_used.append({
                'content': content,
                'sourcefile': doc_name
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
                # Atualizar a URL de acordo com a nova lógica de geração de links
                doc_url = gerar_link_documento(doc_name, selected_index)
                full_response += f"{i+1}. [{doc_name}]({doc_url})\n"

        # Atualiza a resposta final no placeholder
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Função principal do Streamlit
# ... código existente ...

def main():
    st.title("Assistente Virtual Promon")
    logger.info("Iniciando o MakrAI - Assistente Virtual Promon")

    friendly_index_name = st.sidebar.selectbox(
        "Selecione o projeto:",
        options=list(index_mapping.keys())
    )
    selected_index = index_mapping[friendly_index_name]
    logger.info(f"Índice selecionado: {selected_index}")

    print_index_fields(selected_index)

    # Inicializar o histórico de chat se não existir
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Campo de entrada para o usuário
    if prompt := st.chat_input("Digite sua pergunta aqui"):
        handle_chat_prompt(
            prompt,
            aoai_deployment_name,
            aoai_endpoint,
            aoai_key,
            search_endpoint,
            search_key,
            selected_index
        )

    # ... resto do código ...
    # Adicionar disclaimer no rodapé
    st.sidebar.markdown("""""")

    logger.info("Sessão do MakrAI finalizada")

def print_index_fields(index_name):
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_key))
    index = index_client.get_index(index_name)
    logger.debug(f"Campos do índice {index_name}:")
    for field in index.fields:
        logger.debug(f"- {field.name} ({field.type})")

def gerar_link_documento(nome_documento, index_name):
    # Remove o "-7" do nome do documento se existir
    nome_documento = nome_documento.replace('-7.pdf', '.pdf')
    
    # Codifica o nome do documento para URL
    nome_documento_encoded = urllib.parse.quote(nome_documento).replace(' ', '%')
    base_url = "https://aisearchpromon.blob.core.windows.net"
    
    if index_name == "vector-vpak":
        return f"{base_url}/vopak-dp-vetores/PDFs/{nome_documento_encoded}"
    elif index_name == "vector-epotl":
        return f"{base_url}/epotl-dp-vetores/PDFs/{nome_documento_encoded}"
    elif index_name == "vector-rh":
        return f"{base_url}/recursos-humanos/{nome_documento_encoded}"
    elif index_name == "vector-bi":
        return f"{base_url}/bi-im/{nome_documento_encoded}"
    else:
        return f"{base_url}/{index_name}/{nome_documento_encoded}"
        
if __name__ == "__main__":
    main()
