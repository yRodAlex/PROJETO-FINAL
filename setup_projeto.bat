@echo off
echo ====== Configurando o Ambiente Virtual SecureTrust ======

:: Ajuste o caminho abaixo caso o Python esteja instalado em local diferente
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python313\python.exe

:: Cria o ambiente virtual
echo Criando ambiente virtual...
%PYTHON_PATH% -m venv .venv

:: Ativa o ambiente virtual
call .venv\Scripts\activate

:: Atualiza o pip
echo Atualizando o pip...
pip install --upgrade pip

:: Instala os pacotes necessários
echo Instalando dependências...
pip install streamlit pandas numpy scikit-learn==1.6.1 imbalanced-learn==0.13.0 xgboost matplotlib

:: Exibe as versões instaladas
echo ====================================
echo Versão do Python:
python --version

echo Versão do scikit-learn:
python -c "import sklearn; print(sklearn.__version__)"

echo Versão do imbalanced-learn:
python -c "import imblearn; print(imblearn.__version__)"
echo ====================================

echo Ambiente configurado. Execute:
echo .
echo   .venv\Scripts\activate
echo   streamlit run app.py
pause
