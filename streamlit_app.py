import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import yfinance as yf

# Configuração da página
st.set_page_config(
    page_title="XAUUSD Monitor - Didi Aguiar",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #f0b90b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .indicator {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .buy-signal {
        color: #4caf50;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid #4caf50;
        margin: 0.5rem 0;
    }
    .sell-signal {
        color: #f44336;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(244, 67, 54, 0.1);
        border: 1px solid #f44336;
        margin: 0.5rem 0;
    }
    .alert-signal {
        color: #ff9800;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(255, 152, 0, 0.1);
        border: 1px solid #ff9800;
        margin: 0.5rem 0;
    }
    .opportunity-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f5f5f5;
        border-radius: 0.3rem;
        padding: 0.5rem;
        text-align: center;
        flex: 1;
        margin: 0.2rem;
        min-width: 100px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
    /* Otimizações para mobile */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
        }
        .metric-box {
            min-width: 80px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Funções para cálculo de indicadores
def calculate_sma(data, period):
    """Calcula a Média Móvel Simples."""
    return data['Close'].rolling(window=period).mean()

def calculate_didi_index(data):
    """Calcula o Didi Index."""
    # Calcula as médias móveis
    data['SMA3'] = calculate_sma(data, 3)
    data['SMA8'] = calculate_sma(data, 8)
    data['SMA20'] = calculate_sma(data, 20)
    
    # Calcula o Didi Index
    data['didi_fast'] = ((data['SMA3'] / data['SMA8']) - 1) * 100
    data['didi_slow'] = ((data['SMA8'] / data['SMA20']) - 1) * 100
    
    # Identifica cruzamentos (sinais)
    data['didi_signal'] = 0
    for i in range(1, len(data)):
        if data['didi_fast'].iloc[i] > data['didi_slow'].iloc[i] and data['didi_fast'].iloc[i-1] <= data['didi_slow'].iloc[i-1]:
            data.loc[data.index[i], 'didi_signal'] = 1  # Sinal de compra
        elif data['didi_fast'].iloc[i] < data['didi_slow'].iloc[i] and data['didi_fast'].iloc[i-1] >= data['didi_slow'].iloc[i-1]:
            data.loc[data.index[i], 'didi_signal'] = -1  # Sinal de venda
    
    return data

def calculate_adx(data, period=14):
    """Calcula o ADX (Average Directional Index)."""
    # Calcula True Range
    data['high_minus_low'] = data['High'] - data['Low']
    data['high_minus_prev_close'] = abs(data['High'] - data['Close'].shift(1))
    data['low_minus_prev_close'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['high_minus_low', 'high_minus_prev_close', 'low_minus_prev_close']].max(axis=1)
    
    # Calcula +DM e -DM
    data['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
                           np.maximum(data['High'] - data['High'].shift(1), 0), 0)
    data['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
                           np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
    
    # Calcula ATR, +DI e -DI
    data['ATR'] = data['TR'].rolling(window=period).mean()
    data['+DI'] = 100 * (data['+DM'].rolling(window=period).mean() / data['ATR'])
    data['-DI'] = 100 * (data['-DM'].rolling(window=period).mean() / data['ATR'])
    
    # Calcula DX e ADX
    data['DX'] = 100 * (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']))
    data['ADX'] = data['DX'].rolling(window=period).mean()
    
    # Identifica quando o ADX está acima de 32 (tendência forte)
    data['adx_trend'] = np.where(data['ADX'] > 32, 1, 0)
    
    return data

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calcula as Bandas de Bollinger."""
    data['bb_middle'] = data['Close'].rolling(window=period).mean()
    data['bb_std'] = data['Close'].rolling(window=period).std()
    data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
    data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)
    
    # Calcula a largura das bandas
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # Identifica quando há abertura significativa das bandas
    data['bb_width_mean'] = data['bb_width'].rolling(window=period).mean()
    data['bb_significant_opening'] = np.where(data['bb_width'] > data['bb_width_mean'] * 1.1, 1, 0)
    
    return data

def detect_needle(data, index):
    """
    Detecta uma "agulhada" no Didi Index.
    
    Uma "agulhada" ocorre quando as três médias móveis (3, 8 e 20 períodos)
    passam simultaneamente pelo corpo real de um candle.
    """
    if index < 1 or index >= len(data):
        return False, None
    
    # Verifica se há um cruzamento no Didi Index
    if data['didi_signal'].iloc[index] != 0:
        # Verifica se as três médias estão dentro do corpo do candle
        candle_open = data['Open'].iloc[index]
        candle_close = data['Close'].iloc[index]
        candle_min = min(candle_open, candle_close)
        candle_max = max(candle_open, candle_close)
        
        sma3 = data['SMA3'].iloc[index]
        sma8 = data['SMA8'].iloc[index]
        sma20 = data['SMA20'].iloc[index]
        
        # Verifica se todas as médias estão dentro do corpo do candle
        if (candle_min <= sma3 <= candle_max and
            candle_min <= sma8 <= candle_max and
            candle_min <= sma20 <= candle_max):
            
            # Determina o tipo de agulhada (compra ou venda)
            if data['didi_signal'].iloc[index] == 1:
                return True, "compra"
            else:
                return True, "venda"
    
    return False, None

def is_adx_rising(data, index, lookback=3):
    """Verifica se o ADX está crescendo."""
    if index < lookback:
        return False
    
    # Verifica se o ADX atual é maior que o ADX de lookback períodos atrás
    current_adx = data['ADX'].iloc[index]
    previous_adx = data['ADX'].iloc[index - lookback]
    
    return current_adx > previous_adx

def is_price_near_bands(data, index):
    """Verifica se o preço está próximo das Bandas de Bollinger."""
    if index < 0 or index >= len(data):
        return False, None
    
    close = data['Close'].iloc[index]
    upper = data['bb_upper'].iloc[index]
    lower = data['bb_lower'].iloc[index]
    
    # Calcula a distância percentual das bandas
    upper_distance = (upper - close) / close * 100
    lower_distance = (close - lower) / close * 100
    
    # Verifica se o preço está próximo de alguma banda (menos de 0.5% de distância)
    if upper_distance < 0.5:
        return True, "upper"
    elif lower_distance < 0.5:
        return True, "lower"
    
    return False, None

def find_opportunities(data):
    """
    Identifica oportunidades de trading baseadas na metodologia de Didi Aguiar.
    
    Returns:
        list: Lista de oportunidades identificadas
    """
    opportunities = []
    
    # Analisa cada candle para identificar oportunidades
    for i in range(len(data)):
        # Verifica se ocorreu uma agulhada
        is_needle, needle_type = detect_needle(data, i)
        
        if is_needle and needle_type:
            # Verifica se o ADX está acima de 32 e crescendo
            adx_value = data['ADX'].iloc[i]
            adx_trend = data['adx_trend'].iloc[i]
            adx_rising = is_adx_rising(data, i)
            
            # Verifica se as Bandas de Bollinger têm abertura significativa
            bb_significant = data['bb_significant_opening'].iloc[i]
            
            # Verifica se o preço está próximo de alguma banda
            near_band, band_type = is_price_near_bands(data, i)
            
            # Determina o tipo de sinal
            signal_type = "alerta"  # Por padrão, é apenas um alerta
            
            # Se todas as condições forem atendidas, é um sinal completo
            if adx_trend == 1 and adx_rising and bb_significant == 1:
                signal_type = "completo"
            
            # Cria a oportunidade
            opportunity = {
                'timestamp': data.index[i],
                'price': data['Close'].iloc[i],
                'type': needle_type,  # "compra" ou "venda"
                'signal': signal_type,  # "alerta" ou "completo"
                'adx': adx_value,
                'adx_rising': adx_rising,
                'bb_significant': bb_significant == 1,
                'near_band': near_band,
                'band_type': band_type
            }
            
            # Adiciona detalhes específicos para cada tipo de operação
            if needle_type == "compra":
                opportunity['stop_loss'] = data['Low'].iloc[i]
                opportunity['take_profit'] = data['Close'].iloc[i] + (data['Close'].iloc[i] - opportunity['stop_loss']) * 2
            else:  # venda
                opportunity['stop_loss'] = data['High'].iloc[i]
                opportunity['take_profit'] = data['Close'].iloc[i] - (opportunity['stop_loss'] - data['Close'].iloc[i]) * 2
            
            opportunities.append(opportunity)
    
    return opportunities

def get_xauusd_data(interval='15m', days_back=5):
    """
    Obtém dados históricos do XAUUSD.
    
    Args:
        interval (str): Intervalo de tempo dos candles
        days_back (int): Número de dias para trás a considerar
    
    Returns:
        pandas.DataFrame: DataFrame com os dados
    """
    # Calcula a data de início
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    try:
        # Obtém os dados do Yahoo Finance
        data = yf.download('GC=F', start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            st.error("Não foi possível obter dados do XAUUSD. Usando dados simulados.")
            return generate_simulated_data(days=days_back, interval_minutes=int(interval.replace('m', '')))
        
        return data
    
    except Exception as e:
        st.error(f"Erro ao obter dados: {str(e)}. Usando dados simulados.")
        return generate_simulated_data(days=days_back, interval_minutes=int(interval.replace('m', '')))

def generate_simulated_data(days=5, interval_minutes=15):
    """
    Gera dados simulados do XAUUSD.
    
    Args:
        days (int): Número de dias para simular
        interval_minutes (int): Intervalo em minutos entre os candles
    
    Returns:
        pandas.DataFrame: DataFrame com os dados simulados
    """
    # Calcula o número de candles
    candles_per_day = 24 * 60 // interval_minutes
    num_candles = days * candles_per_day
    
    # Gera timestamps
    end_time = datetime.datetime.now()
    timestamps = [end_time - datetime.timedelta(minutes=i*interval_minutes) for i in range(num_candles)]
    timestamps.reverse()
    
    # Gera preços simulados
    base_price = 2000.0  # Preço base do ouro
    volatility = 0.002  # Volatilidade (0.2%)
    trend = 0.0001  # Tendência leve de alta
    
    prices = [base_price]
    for i in range(1, num_candles):
        # Adiciona um componente de tendência e um componente aleatório
        change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Gera candles OHLC
    data = []
    for i in range(num_candles):
        close = prices[i]
        # Gera Open, High, Low com base no Close
        open_price = close * (1 + np.random.normal(0, volatility/2))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility/2)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility/2)))
        
        # Adiciona alguns padrões de "agulhada" aleatoriamente
        if i > 20 and np.random.random() < 0.05:  # 5% de chance de criar uma agulhada
            # Decide se é uma agulhada de compra ou venda
            if np.random.random() < 0.5:  # Compra
                open_price = low * 1.001
                close = high * 0.999
            else:  # Venda
                open_price = high * 0.999
                close = low * 1.001
        
        # Volume simulado
        volume = np.random.randint(1000, 5000)
        
        data.append([open_price, high, low, close, volume])
    
    # Cria o DataFrame
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=timestamps)
    
    return df

def plot_chart(data, opportunities):
    """
    Cria um gráfico interativo com os dados e oportunidades.
    
    Args:
        data (pandas.DataFrame): DataFrame com os dados
        opportunities (list): Lista de oportunidades identificadas
    
    Returns:
        plotly.graph_objects.Figure: Figura do gráfico
    """
    # Cria subplots: gráfico de preço, Didi Index e ADX
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("XAUUSD", "Didi Index", "ADX"))
    
    # Adiciona o gráfico de candles
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="XAUUSD"
    ), row=1, col=1)
    
    # Adiciona as médias móveis
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA3'],
        name="SMA 3",
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA8'],
        name="SMA 8",
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA20'],
        name="SMA 20",
        line=dict(color='red', width=1)
    ), row=1, col=1)
    
    # Adiciona as Bandas de Bollinger
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['bb_upper'],
        name="BB Superior",
        line=dict(color='gray', width=1, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['bb_lower'],
        name="BB Inferior",
        line=dict(color='gray', width=1, dash='dash')
    ), row=1, col=1)
    
    # Adiciona o Didi Index
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['didi_fast'],
        name="Didi Fast",
        line=dict(color='blue', width=1.5)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['didi_slow'],
        name="Didi Slow",
        line=dict(color='red', width=1.5)
    ), row=2, col=1)
    
    # Adiciona uma linha horizontal em 0 para o Didi Index
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=0,
        x1=data.index[-1],
        y1=0,
        line=dict(color="black", width=1, dash="dot"),
        row=2, col=1
    )
    
    # Adiciona o ADX
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['ADX'],
        name="ADX",
        line=dict(color='purple', width=1.5)
    ), row=3, col=1)
    
    # Adiciona uma linha horizontal em 32 para o ADX
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=32,
        x1=data.index[-1],
        y1=32,
        line=dict(color="red", width=1, dash="dot"),
        row=3, col=1
    )
    
    # Marca as oportunidades no gráfico
    for opp in opportunities:
        # Cor e símbolo com base no tipo de operação e sinal
        if opp['type'] == 'compra':
            marker_color = 'green'
            marker_symbol = 'triangle-up'
        else:  # venda
            marker_color = 'red'
            marker_symbol = 'triangle-down'
        
        # Tamanho com base no tipo de sinal
        marker_size = 12 if opp['signal'] == 'completo' else 8
        
        # Adiciona o marcador
        fig.add_trace(go.Scatter(
            x=[opp['timestamp']],
            y=[opp['price']],
            mode='markers',
            marker=dict(
                color=marker_color,
                size=marker_size,
                symbol=marker_symbol,
                line=dict(color='black', width=1)
            ),
            name=f"{opp['type'].capitalize()} ({opp['signal'].capitalize()})",
            showlegend=False
        ), row=1, col=1)
    
    # Configurações de layout
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configurações dos eixos Y
    fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Didi Index", row=2, col=1)
    fig.update_yaxes(title_text="ADX", row=3, col=1)
    
    return fig

def display_opportunity_card(opportunity):
    """
    Exibe um card com os detalhes da oportunidade.
    
    Args:
        opportunity (dict): Dicionário com os detalhes da oportunidade
    """
    # Determina as classes CSS com base no tipo de operação e sinal
    if opportunity['type'] == 'compra':
        header_class = 'buy-signal'
        header_text = 'COMPRA'
    else:  # venda
        header_class = 'sell-signal'
        header_text = 'VENDA'
    
    if opportunity['signal'] == 'completo':
        signal_text = 'SINAL COMPLETO'
    else:
        signal_text = 'ALERTA'
        header_class = 'alert-signal'
    
    # Formata a data/hora
    timestamp = opportunity['timestamp'].strftime('%d/%m/%Y %H:%M')
    
    # Exibe o card
    st.markdown(f"""
    <div class="opportunity-card">
        <div class="{header_class}">
            {header_text} - {signal_text} - {timestamp}
        </div>
        
        <div class="metric-container">
            <div class="metric-box">
                <div class="metric-label">Preço</div>
                <div class="metric-value">${opportunity['price']:.2f}</div>
            </div>
            
            <div class="metric-box">
                <div class="metric-label">ADX</div>
                <div class="metric-value">{opportunity['adx']:.2f}</div>
            </div>
            
            <div class="metric-box">
                <div class="metric-label">Stop Loss</div>
                <div class="metric-value">${opportunity['stop_loss']:.2f}</div>
            </div>
            
            <div class="metric-box">
                <div class="metric-label">Take Profit</div>
                <div class="metric-value">${opportunity['take_profit']:.2f}</div>
            </div>
        </div>
        
        <div>
            <p>ADX: {'↗️ Crescendo' if opportunity['adx_rising'] else '➡️ Estável/Caindo'}</p>
            <p>Bandas de Bollinger: {'✅ Abertura Significativa' if opportunity['bb_significant'] else '❌ Abertura Normal'}</p>
            <p>Risco/Retorno: 1:{((opportunity['take_profit'] - opportunity['price']) / (opportunity['price'] - opportunity['stop_loss'])):.1f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Interface principal
def main():
    # Cabeçalho
    st.markdown('<h1 class="main-header">XAUUSD Monitor - Metodologia Didi Aguiar</h1>', unsafe_allow_html=True)
    
    # Sidebar para configurações
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Configurações</h2>', unsafe_allow_html=True)
        
        # Intervalo de tempo
        interval = st.selectbox(
            "Intervalo de tempo",
            options=["1m", "5m", "15m", "30m", "60m", "1d"],
            index=2  # 15m por padrão
        )
        
        # Dias para trás
        days_back = st.slider(
            "Dias para análise",
            min_value=1,
            max_value=30,
            value=5
        )
        
        # Botão para atualizar dados
        if st.button("Atualizar Dados"):
            st.session_state.update_time = time.time()
            st.session_state.update_counter = st.session_state.get('update_counter', 0) + 1
    
    # Inicializa o contador de atualizações se não existir
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
        st.session_state.update_time = time.time()
    
    # Exibe o status da última atualização
    last_update = datetime.datetime.fromtimestamp(st.session_state.update_time).strftime('%d/%m/%Y %H:%M:%S') if hasattr(st.session_state, 'update_time') else "Nunca"
    st.markdown(f"<p>Última atualização: {last_update}</p>", unsafe_allow_html=True)
    
    # Obtém os dados
    with st.spinner("Carregando dados do XAUUSD..."):
        data = get_xauusd_data(interval=interval, days_back=days_back)
        
        # Calcula os indicadores
        data = calculate_didi_index(data)
        data = calculate_adx(data)
        data = calculate_bollinger_bands(data)
        
        # Identifica oportunidades
        opportunities = find_opportunities(data)
    
    # Exibe o gráfico
    st.markdown('<h2 class="sub-header">Gráfico XAUUSD</h2>', unsafe_allow_html=True)
    fig = plot_chart(data, opportunities)
    st.plotly_chart(fig, use_container_width=True)
    
    # Exibe as oportunidades identificadas
    st.markdown('<h2 class="sub-header">Oportunidades Identificadas</h2>', unsafe_allow_html=True)
    
    # Filtra as oportunidades mais recentes (últimas 24 horas)
    recent_time = datetime.datetime.now() - datetime.timedelta(hours=24)
    recent_opportunities = [opp for opp in opportunities if opp['timestamp'] > recent_time]
    
    # Exibe o contador de oportunidades
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-box">
            <div class="metric-label">Total de Oportunidades</div>
            <div class="metric-value">{len(opportunities)}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Últimas 24 horas</div>
            <div class="metric-value">{len(recent_opportunities)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Exibe as oportunidades recentes
    if recent_opportunities:
        for opp in sorted(recent_opportunities, key=lambda x: x['timestamp'], reverse=True):
            display_opportunity_card(opp)
    else:
        st.info("Nenhuma oportunidade recente identificada nas últimas 24 horas.")
    
    # Exibe informações sobre a metodologia
    with st.expander("Sobre a Metodologia de Didi Aguiar"):
        st.markdown("""
        ### Metodologia de Didi Aguiar

        A metodologia de Didi Aguiar, desenvolvida pelo analista brasileiro Odir Aguiar (Didi), baseia-se na identificação de pontos de reversão no mercado através de um conjunto de indicadores técnicos:

        1. **Didi Index**: Indicador principal que utiliza três médias móveis (3, 8 e 20 períodos) para identificar "agulhadas" - momentos de reversão no mercado.

        2. **ADX (Average Directional Index)**: Utilizado para confirmar a força da tendência, com sinal positivo quando está acima do nível 32 e crescendo.

        3. **Bandas de Bollinger**: Utilizadas para confirmar volatilidade, com sinal positivo quando há abertura significativa das bandas.

        Um sinal completo ocorre quando todos estes elementos se alinham simultaneamente, indicando uma oportunidade de alta probabilidade de sucesso.
        """)
    
    # Rodapé
    st.markdown('<div class="footer">XAUUSD Monitor - Versão Web Simplificada - Baseado na Metodologia de Didi Aguiar</div>', unsafe_allow_html=True)

# Executa a aplicação
if __name__ == "__main__":
    main()
