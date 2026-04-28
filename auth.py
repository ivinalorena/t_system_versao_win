"""
Módulo de autenticação para TaguchiApp
Gerencia login, registro e segurança com bcrypt
"""
import bcrypt
import streamlit as st
from database import (
    init_db,
    user_exists,
    create_user,
    get_user,
    update_last_login,
    log_action,
)


def hash_password(password: str) -> str:
    """Hash de uma senha usando bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verifica se uma senha corresponde ao hash"""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def register_user(username: str, email: str, password: str, password_confirm: str) -> tuple[bool, str]:
    """
    Registra um novo usuário
    Retorna (sucesso, mensagem)
    """
    # Validações
    if not username or len(username) < 3:
        return False, "❌ Nome de usuário deve ter pelo menos 3 caracteres"
    
    if not email or "@" not in email:
        return False, "❌ Email inválido"
    
    if not password or len(password) < 6:
        return False, "❌ Senha deve ter pelo menos 6 caracteres"
    
    if password != password_confirm:
        return False, "❌ As senhas não coincidem"
    
    if user_exists(username):
        return False, "❌ Este nome de usuário já existe"
    
    # Cria o usuário
    password_hash = hash_password(password)
    if create_user(username, email, password_hash):
        log_action(None, "user_registered", f"username: {username}, email: {email}")
        return True, "✅ Usuário registrado com sucesso! Faça login agora."
    else:
        return False, "❌ Erro ao registrar usuário. Verifique se o email já não está em uso."


def login_user(username: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Autentica um usuário
    Retorna (sucesso, mensagem, dados_usuario)
    """
    user = get_user(username)
    
    if not user:
        return False, "❌ Usuário ou senha incorretos", None
    
    if not user["is_active"]:
        return False, "❌ Conta desativada", None
    
    if not verify_password(password, user["password_hash"]):
        return False, "❌ Usuário ou senha incorretos", None
    
    # Atualiza último login
    update_last_login(user["id"])
    log_action(user["id"], "login", f"username: {username}")
    
    return True, "✅ Login realizado com sucesso!", user


def render_login_page():
    """
    Renderiza a página de login/registro
    Retorna True se o usuário está autenticado
    """
    # Inicializa o banco de dados
    init_db()
    
    # Verifica se já está logado
    if "user_id" in st.session_state:
        return True
    
    # Layout da página de login
    st.set_page_config(page_title="TaguchiApp - Login", layout="centered")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# TaguchiApp")
        st.markdown("### Planejamento e Análise Experimental")
        st.markdown("---")
        
        # Abas para Login e Registro
        tab_login, tab_register = st.tabs([" Login", "Registrar "])
        
        with tab_login:
            st.markdown("### Entre na sua conta")
            
            username = st.text_input("Usuário", key="login_username")
            password = st.text_input("Senha", type="password", key="login_password")
            
            if st.button("Entrar", use_container_width=True, type="primary"):
                if username and password:
                    success, message, user = login_user(username, password)
                    
                    if success:
                        st.session_state.user_id = user["id"]
                        st.session_state.username = user["username"]
                        st.session_state.email = user["email"]
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning(" Preencha usuário e senha")
            
            st.markdown("---")
            st.markdown(
                "<div style='text-align: center; color: #666; font-size: 12px;'>"
                "Versão v1.2026 | Sistema Taguchi<br>"
                "Desenvolvido com Streamlit"
                "</div>",
                unsafe_allow_html=True
            )
        
        with tab_register:
            st.markdown("### Crie uma nova conta")
            
            new_username = st.text_input("Nome de usuário", key="register_username")
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Senha", type="password", key="register_password")
            new_password_confirm = st.text_input(
                "Confirme a senha",
                type="password",
                key="register_password_confirm"
            )
            
            if st.button("Registrar", use_container_width=True, type="primary"):
                success, message = register_user(new_username, new_email, new_password, new_password_confirm)
                
                if success:
                    st.success(message)
                    st.info("Agora faça login com suas credenciais na aba anterior")
                else:
                    st.error(message)
            
            st.markdown("---")
            st.markdown(
                "<div style='text-align: center; color: #666; font-size: 12px;'>"
                "Versão v1.2026 | Sistema Taguchi<br>"
                "Desenvolvido com Streamlit"
                "</div>",
                unsafe_allow_html=True
            )
    
    return False


def render_logout_button():
    """Renderiza o botão de logout na barra lateral"""
    with st.sidebar:
        st.markdown(f" Bem-vindo(a), **{st.session_state.username}**")
        #st.markdown(f" {st.session_state.email}")
        st.markdown("---")
        
        if st.button("Logout", use_container_width=True):
            log_action(st.session_state.user_id, "logout", f"username: {st.session_state.username}")
            # Limpa a sessão
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
