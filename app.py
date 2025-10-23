import streamlit as st
from pathlib import Path

# LangChain 관련 라이브러리 (ChromaDB 최종본)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma # ChromaDB 사용
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# --- 페이지 설정 ---
st.set_page_config(
    page_title="📄 Chat with Document",
    page_icon="📄",
    layout="wide"
)
st.title("📄 문서 기반 채팅 애플리케이션")

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input(
        "OpenAI API Key를 입력하세요",
        type="password",
        help="API 키는 OpenAI 웹사이트에서 발급받을 수 있습니다."
    )
    uploaded_file = st.file_uploader(
        "분석할 문서를 업로드하세요 (.txt)",
        type=["txt"]
    )
    st.markdown("---")
    st.markdown(
        "❤️ [GitHub Repository](https://github.com/jj-prog3/streamlit-rag)"
    )

# --- 세션 상태 초기화 ---
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="안녕하세요! 분석할 문서를 업로드하고 질문을 시작하세요.")
    ]

# --- 핵심 로직: 파일 처리 및 체인 생성 ---
@st.cache_resource(show_spinner="⏳ 문서를 처리하고 있습니다...")
def process_file_and_create_chain(_api_key, _uploaded_file):
    try:
        file_content = _uploaded_file.getvalue().decode("utf-8")
        docs = [Document(page_content=file_content)]

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        split_docs = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=_api_key)
        
        # Chroma를 사용하여 벡터 저장소 생성
        vectorstore = Chroma.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 주어진 문서의 내용을 기반으로 질문에 답변하는 AI 어시스턴트입니다.\n문서 내용에 없는 정보는 답변하지 말고, 모른다고 솔직하게 말하세요.\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(),
                "history": lambda x: st.session_state.get('messages', [])
            }
            | prompt
            | ChatOpenAI(temperature=0.1, max_tokens=1024, openai_api_key=_api_key)
            | StrOutputParser()
        )
        return rag_chain
    
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
        return None

# --- 메인 화면 로직 ---
if api_key and uploaded_file:
    # 파일이 변경되면 체인을 다시 생성
    st.session_state.chain = process_file_and_create_chain(api_key, uploaded_file)
    
    if "new_file" not in st.session_state or st.session_state.new_file != uploaded_file.name:
        st.session_state.new_file = uploaded_file.name
        st.session_state.messages = [
            AIMessage(content=f"'{uploaded_file.name}'에 대해 질문해주세요.")
        ]
else:
    st.info("👈 사이드바에서 OpenAI API Key를 입력하고 문서를 업로드해주세요.")

# 채팅 기록 표시
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

# 사용자 입력 처리
if user_query := st.chat_input("문서에 대해 질문을 입력하세요..."):
    if not st.session_state.chain:
        st.warning("먼저 문서를 업로드하고 처리해야 합니다.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)

        with st.spinner("답변을 생성 중입니다..."):
            response = st.session_state.chain.invoke(user_query)
            st.session_state.messages.append(AIMessage(content=response))
            st.chat_message("assistant").write(response)