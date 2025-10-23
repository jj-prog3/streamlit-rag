import streamlit as st
from pathlib import Path

# LangChain 관련 라이브러리 (버전 고정 최종본)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
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
        "❤️ [GitHub Repository](https://github.com/your-username/your-repo)" # 본인의 리포지토리 주소로 변경하세요
    )

# --- 세션 상태 초기화 ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="안녕하세요! 분석할 문서를 업로드하고 질문을 시작하세요.")
        ]
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None

initialize_session_state()

# --- 핵심 로직: 파일 처리 및 체인 생성 ---
def process_file_and_create_chain(api_key, uploaded_file):
    try:
        llm = ChatOpenAI(temperature=0.1, max_tokens=1024, openai_api_key=api_key)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        file_content = uploaded_file.getvalue().decode("utf-8")
        raw_doc = [Document(page_content=file_content)]

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n", chunk_size=300, chunk_overlap=50
        )
        docs = splitter.split_documents(raw_doc)

        cache_dir = LocalFileStore(f"./.cache/{uploaded_file.name}")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 주어진 문서의 내용을 기반으로 질문에 답변하는 AI 어시스턴트입니다.\n문서 내용에 없는 정보는 답변하지 말고, 모른다고 솔직하게 말하세요.\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def get_history(session_messages):
            return [msg for msg in session_messages if not isinstance(msg, AIMessage) or len(session_messages) == 1]

        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(),
                "history": lambda x: st.session_state.get('messages', [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.session_state.chain = rag_chain
        st.session_state.processed_file_name = uploaded_file.name
        return True
    
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
        return False

# --- 메인 화면 로직 ---
if not api_key:
    st.info("👈 사이드바에서 OpenAI API Key를 입력해주세요.")
elif not uploaded_file:
    st.info("👈 사이드바에서 분석할 문서를 업로드해주세요.")
else:
    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("⏳ 문서를 처리하고 있습니다..."):
            if process_file_and_create_chain(api_key, uploaded_file):
                st.success(f"'{uploaded_file.name}' 문서 처리가 완료되었습니다!")
                st.session_state.messages = [
                    AIMessage(content=f"'{uploaded_file.name}'에 대해 질문해주세요.")
                ]
    
    # 채팅 기록 표시
    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        st.chat_message(role).write(msg.content)

    # 사용자 입력 처리
    if user_query := st.chat_input("문서에 대해 질문을 입력하세요..."):
        if st.session_state.chain:
            st.session_state.messages.append(HumanMessage(content=user_query))
            st.chat_message("user").write(user_query)

            with st.spinner("답변을 생성 중입니다..."):
                response = st.session_state.chain.invoke(user_query)
                st.session_state.messages.append(AIMessage(content=response))
                st.chat_message("assistant").write(response)
        else:
            st.warning("문서가 아직 처리되지 않았습니다. 잠시만 기다려주세요.")