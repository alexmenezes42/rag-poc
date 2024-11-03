import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import shutil

# Configuração da página
st.set_page_config(page_title="POC - Chat Bot", layout="wide")

# Título do aplicativo
st.title('Seja vítima da POC')


# Sessão para armazenar o histórico da conversa
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Entrada da chave API da OpenAI na barra lateral
# api_key = st.sidebar.text_input("Insira sua Chave API da OpenAI:", type="password")

# # Verificação da chave API
# if not api_key:
#     st.sidebar.warning("Por favor, insira sua chave API da OpenAI para continuar.")
#     st.stop()

api_key = st.secrets["api_key"]
# Função para extrair texto dos PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Função para dividir o texto em pedaços
def get_text_chunks(text):
    # Ajuste os parâmetros conforme necessário
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

# Função para criar a loja vetorial usando embeddings da OpenAI
def get_vector_store(text_chunks, api_key):
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Função para criar a cadeia de conversação
def get_conversational_chain(api_key):
    prompt_template = """
Responda à pergunta de forma detalhada com base no contexto fornecido. Certifique-se de fornecer todos os detalhes. Se a resposta não estiver no contexto fornecido, tente responder da forma mais adequada.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatOpenAI(model_name="gpt-4", temperature=0.3, openai_api_key=api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Função para processar a pergunta do usuário
def generate_response(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if not os.path.exists("faiss_index"):
        st.error("Por favor, processe os documentos primeiro.")
        return None
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Função principal
def main():
    # Seção de configurações na barra lateral
    st.sidebar.subheader("Sobre a empresa (orientações)")
    company_info = st.sidebar.text_area(
        "Insira aqui as informações ou orientações sobre a empresa.",
        """A Findor oferece uma plataforma inovadora que utiliza inteligência artificial para transformar as interações empresariais através de “personagens” virtuais capazes de realizar negócios por meio de conversas. Esses personagens são projetados para aprender sobre a empresa, realizar transações, qualificar e atender clientes de maneira rápida e eficiente, proporcionando uma experiência de alto nível para o usuário.

### Funcionalidades da Findor

1. **Personalização e Treinamento de Personagens**:
   - Configure a personalidade, o tom de voz e as habilidades do personagem, ajustando como ele interage com os clientes.
   - Defina o tempo de resposta e diretrizes específicas para que ele compreenda e represente os processos da empresa de forma precisa.

2. **Gestão de Conversas**:
   - Permite gerenciar e monitorar todas as interações em tempo real, possibilitando a interrupção das respostas automáticas caso necessário.
   - Inclui filtros inteligentes e feedbacks que ajudam a aprimorar o aprendizado do personagem.

3. **Automação de Campanhas e CRM**:
   - Facilita o acompanhamento de conversas abandonadas e a atualização automática de leads no CRM da empresa.
   - Integra planilhas, como Excel ou Google Sheets, para melhor organização de dados e gestão de campanhas.

4. **Análise de Resultados**:
   - Oferece métricas personalizadas e visuais organizáveis para que a empresa acompanhe o desempenho em tempo real.
   - Assegura uma visão clara sobre os principais KPIs, permitindo análises aprofundadas de engajamento e conversão.

5. **Integrações Completas**:
   - A Findor se conecta a diversas plataformas, incluindo CRMs e ferramentas de comunicação, facilitando a integração dos personagens aos canais de comunicação preferidos.

### Benefícios Operacionais e Redução de Custos
Estudos de caso apresentados indicam que a Findor ajuda empresas a reduzir custos em mais de 60% e a aumentar a satisfação dos clientes. Exemplo de resultados:
   - Para bancos digitais, a plataforma diminuiu o custo em pessoal e ferramentas, enquanto aumentou a taxa de aprovação e reduziu o tempo médio de resolução.
   - No caso de instituições de ensino, elevou o NPS e reduziu a taxa de abandono, resultando em maior engajamento.
   - Em empresas de assessoria de investimentos, a Findor foi capaz de reduzir o custo por reunião em até 70%, triplicando a conversão de leads.

### Orientações para Utilização
Para aproveitar ao máximo a Findor, a empresa deve:
   - **Definir claramente o perfil e o tom de voz do personagem** para que ele esteja alinhado com a cultura e os valores da organização.
   - **Monitorar e ajustar o aprendizado continuamente** para garantir que os personagens respondam adequadamente às necessidades dos clientes.
   - **Integrar a Findor aos sistemas existentes**, como CRM e plataformas de comunicação, para centralizar dados e otimizar o processo de atendimento.

Com a Findor, sua empresa estará equipada para atender às demandas da nova era digital, otimizando operações e ampliando a eficiência através de interações baseadas em inteligência artificial.""",
        height=130
    )

    # Upload de PDFs na barra lateral
    st.sidebar.subheader("Documentos")
    pdf_docs = st.sidebar.file_uploader(
        "Carregue seus Arquivos PDF",
        accept_multiple_files=True,
        type=['pdf'],
        key="pdf_uploader"
    )

    if not pdf_docs:
        default_pdf_path = 'findor.pdf'  # Substitua pelo caminho do seu arquivo PDF padrão
        if os.path.exists(default_pdf_path):
            with open(default_pdf_path, 'rb') as f:
                pdf_docs = [f]

    # Seção de FAQ na barra lateral
    st.sidebar.subheader("Seção de FAQ")
    num_faq = st.sidebar.number_input(
        "Informe a quantidade de itens",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        key="num_faq"
    )
    faq_pairs = []
    for i in range(int(num_faq)):
        st.sidebar.markdown(f"**FAQ {i+1}**")
        # Definimos valores padrão para as duas primeiras FAQs
        if i == 0:
            question_default = "Quais modelos vocês utilizam?"
            answer_default = "Construímos um framework que se conecta aos principais provedores de IA Generativa (OpenAI, Gemini, Claude, Llama). Cada provedor é mais eficiente para determinados tipos de tarefas. Abstraímos essa complexidade para você, cuidando das atualizações constantes na IA, permitindo que você se concentre no seu negócio."
        elif i == 1:
            question_default = "Como funciona o processo de implantação? O processo de implantação é demorado? COnsigo integrar com o meu CRM?"
            answer_default = "Realizamos uma reunião inicial para demonstrar a plataforma e entender suas necessidades. Durante essa reunião, a versão “Júnior” do seu personagem já estará pronta para testes. Em seguida, entramos em um ciclo de feedback e testes para evoluí-lo até que esteja pronto para interagir com seus clientes ou equipes internas. Todas as integrações com CRMs ou canais são de nossa responsabilidade."
        else:
            question_default = ""
            answer_default = ""
        question = st.sidebar.text_input(
            f"Pergunta {i+1}:", value=question_default, key=f"faq_question_{i+1}"
        )
        answer = st.sidebar.text_area(
            f"Resposta {i+1}:", value=answer_default, key=f"faq_answer_{i+1}", height=100
        )
        if question and answer:
            faq_pairs.append({"question": question, "answer": answer})

    # Botão de submissão na barra lateral
    if st.sidebar.button('Enviar & Processar'):
        with st.spinner("Processando..."):
            # Extrair texto dos PDFs
            raw_text = get_pdf_text(pdf_docs) if pdf_docs else ""

            # Adicionar informações da empresa
            if company_info:
                raw_text += "\n---\n**Sobre a Empresa:**\n" + company_info + "\n---\n"

            # Adicionar FAQs com delimitadores claros
            if faq_pairs:
                raw_text += "\n---\n**Seção de FAQ:**\n"
                for faq in faq_pairs:
                    raw_text += f"**Pergunta:** {faq['question']}\n**Resposta:** {faq['answer']}\n\n"
                raw_text += "---\n"

            # Dividir o texto em chunks
            text_chunks = get_text_chunks(raw_text)

            # Criar a loja vetorial
            get_vector_store(text_chunks, api_key)

            st.sidebar.success("Processamento Concluído!")

    st.markdown("---")


    # Create a container for the chat history
    chat_container = st.container()

    with chat_container:
        # Display each message
        for message in st.session_state.messages:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])

    # Campo de entrada para a pergunta do usuário
    # Use st.chat_input for better chat interface
    user_question = st.chat_input("Digite sua mensagem aqui:")

    if user_question:
        with st.spinner("Gerando resposta..."):
            response = generate_response(user_question, api_key)
            if response:
                # Armazenar a mensagem do usuário e a resposta do bot
                st.session_state.messages.append({"role": "user", "content": user_question})
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Display the new messages
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    with st.chat_message("assistant"):
                        st.markdown(response)

# Execução do aplicativo
if __name__ == "__main__":
    main()
