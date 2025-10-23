import streamlit as st
from pathlib import Path
import copy

# LangChain 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

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

    # 1. 사용자 OpenAI API 키 입력
    api_key = st.text_input(
        "OpenAI API Key를 입력하세요",
        type="password",
        help="API 키는 OpenAI 웹사이트에서 발급받을 수 있습니다."
    )

    # 2. 문서 업로드
    uploaded_file = st.file_uploader(
        "분석할 문서를 업로드하세요 (.txt)",
        type=["txt"]
    )

    # 3. GitHub 리포지토리 링크
    st.markdown("---")
    st.markdown(
        "❤️ [GitHub 리포지토리](https://github.com/your-username/your-repo)"
    )


# --- 세션 상태 초기화 ---
def initialize_session_state():
    """세션 상태에 필요한 키들을 초기화합니다."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="안녕하세요! 분석할 문서를 업로드하고 질문을 시작하세요.")
        ]
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None

initialize_session_state()


# --- 핵심 로직: 파일 처리 및 체인 생성 ---
def process_file_and_create_chain(api_key, uploaded_file):
    """업로드된 파일을 처리하고 RAG 체인을 생성하여 세션 상태에 저장합니다."""
    try:
        # LLM 및 임베딩 모델 초기화
        llm = ChatOpenAI(temperature=0.1, max_tokens=1024, openai_api_key=api_key)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # 업로드된 파일 읽기
        file_content = uploaded_file.getvalue().decode("utf-8")
        
        # 문서를 Document 객체로 변환
        raw_doc = [Document(page_content=file_content)]

        # 텍스트 분할
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=300,
            chunk_overlap=50,
        )
        docs = splitter.split_documents(raw_doc)

        # 캐시 설정 및 임베딩
        cache_dir = LocalFileStore("./.cache/")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        
        # 벡터 저장소 생성 (FAISS)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 2개의 관련 청크 검색

        # 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 주어진 문서의 내용을 기반으로 질문에 답변하는 유용한 AI 어시스턴트입니다.\n"
                    "문서 내용에 없는 정보는 답변하지 말고, 모른다고 솔직하게 말하세요.\n\n"
                    "Context:\n{context}"
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # RAG 체인 구성
        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
                "history": lambda x: st.session_state.get('history', [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 생성된 retriever와 chain을 세션 상태에 저장
        st.session_state.retriever = retriever
        st.session_state.chain = chain
        st.session_state.processed_file_name = uploaded_file.name
        
        return True
    
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
        return False

# --- 메인 화면 로직 ---

# API 키와 파일이 모두 준비되었는지 확인
if not api_key:
    st.info("👈 사이드바에서 OpenAI API Key를 입력해주세요.")
elif not uploaded_file:
    st.info("👈 사이드바에서 분석할 문서를 업로드해주세요.")
else:
    # 파일이 변경되었는지 확인하고, 변경되었다면 새로 처리
    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("⏳ 문서를 처리하고 있습니다..."):
            if process_file_and_create_chain(api_key, uploaded_file):
                st.success(f"'{uploaded_file.name}' 문서 처리가 완료되었습니다!")
                # 새 문서가 처리되었으므로 채팅 기록 초기화
                st.session_state.messages = [
                    AIMessage(content=f"'{uploaded_file.name}'에 대해 질문해주세요.")
                ]


    # 채팅 기록 표시
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    # 사용자 입력 처리
    if user_query := st.chat_input("문서에 대해 질문을 입력하세요..."):
        # 체인이 준비되었는지 확인
        if st.session_state.chain:
            # 사용자 메시지 추가 및 표시
            st.session_state.messages.append(HumanMessage(content=user_query))
            st.chat_message("user").write(user_query)

            with st.spinner("답변을 생성 중입니다..."):
                # RAG 체인 실행
                response = st.session_state.chain.invoke(
                    user_query,
                    # LangChain Expression Language (LCEL)을 사용하면, 
                    # history는 prompt 템플릿 내에서 lambda 함수를 통해 동적으로 전달됩니다.
                )

                # AI 응답 추가 및 표시
                st.session_state.messages.append(AIMessage(content=response))
                st.chat_message("assistant").write(response)
                
                # 다음 호출을 위해 history 업데이트
                st.session_state['history'] = st.session_state.messages[-2:] # 최근 질문과 답변만 히스토리로 관리

        else:
            st.warning("문서가 아직 처리되지 않았습니다. 잠시만 기다려주세요.")