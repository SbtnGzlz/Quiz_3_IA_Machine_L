// ============================================================
// CONFIGURACIÓN DEL SCALER (valores extraídos del entrenamiento)
// ============================================================
const scalerMean = [51.96666666666667, 134.84761904761905, 224.21428571428572, 0.5, 0.4857142857142857];
const scalerStd = [15.844597496978894, 25.26423444219344, 45.398518810684585, 0.5, 0.4997958767010258];

// ============================================================
// PARÁMETROS DE LA REGRESIÓN LOGÍSTICA
// ============================================================
let logisticParams = null;

// ============================================================
// PESOS DE LA RED NEURONAL
// ============================================================
let nnWeights = null;
let nnModel = null;

// ============================================================
// CARGAR MODELOS AL INICIO
// ============================================================
window.addEventListener('load', async () => {
    console.log("🚀 Iniciando carga de modelos...");
    document.getElementById('loading').style.display = 'block';
    
    try {
        // 1. Cargar parámetros de regresión logística
        const logResponse = await fetch('model/logistic_params.json');
        logisticParams = await logResponse.json();
        console.log("✅ Regresión Logística cargada");
        
        // 2. Cargar pesos de la red neuronal
        const nnResponse = await fetch('model/nn_weights.json');
        nnWeights = await nnResponse.json();
        console.log("✅ Pesos de Red Neuronal cargados");
        
        // 3. Crear modelo de red neuronal en TensorFlow.js
        nnModel = crearModeloNN(nnWeights);
        console.log("✅ Red Neuronal construida");
        
        // Habilitar botón de predicción
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').textContent = "Calcular Riesgo";
        document.getElementById('loading').style.display = 'none';
        
        console.log("🎉 ¡Todos los modelos cargados correctamente!");
        
    } catch (error) {
        console.error("❌ Error cargando modelos:", error);
        alert("Error al cargar los modelos. Verifica que los archivos JSON estén en la carpeta 'model'.");
        document.getElementById('loading').innerHTML = 
            '<p style="color: red;">Error al cargar. Revisa la consola (F12).</p>';
    }
});

// ============================================================
// CREAR MODELO DE RED NEURONAL CON PESOS CARGADOS
// ============================================================
function crearModeloNN(weights) {
    // Crear modelo Sequential
    const model = tf.sequential();
    
    // Capa 1: Dense con 8 neuronas, activación ReLU
    model.add(tf.layers.dense({
        units: 8,
        activation: 'relu',
        inputShape: [5],
        weights: [
            tf.tensor2d(weights.hidden1_kernel),  // kernel [5, 8]
            tf.tensor1d(weights.hidden1_bias)     // bias [8]
        ]
    }));
    
    // Capa 2: Dense con 4 neuronas, activación ReLU
    model.add(tf.layers.dense({
        units: 4,
        activation: 'relu',
        weights: [
            tf.tensor2d(weights.hidden2_kernel),  // kernel [8, 4]
            tf.tensor1d(weights.hidden2_bias)     // bias [4]
        ]
    }));
    
    // Capa 3: Dense con 1 neurona, activación Sigmoid
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        weights: [
            tf.tensor2d(weights.output_kernel),   // kernel [4, 1]
            tf.tensor1d(weights.output_bias)      // bias [1]
        ]
    }));
    
    console.log("📊 Resumen del modelo creado:");
    model.summary();
    
    return model;
}

// ============================================================
// FUNCIÓN DE ESCALADO (StandardScaler)
// ============================================================
function escalar(valores) {
    return valores.map((v, i) => {
        const num = Number(v);
        if (!isFinite(num)) {
            console.error(`Valor no válido en posición ${i}:`, v);
            return NaN;
        }
        return (num - scalerMean[i]) / scalerStd[i];
    });
}

// ============================================================
// PREDICCIÓN CON REGRESIÓN LOGÍSTICA
// ============================================================
function predecirLogistica(valoresEscalados) {
    // Calcular z = w1*x1 + w2*x2 + ... + wn*xn + b
    let z = logisticParams.intercept;
    
    for (let i = 0; i < valoresEscalados.length; i++) {
        z += logisticParams.coefficients[i] * valoresEscalados[i];
    }
    
    // Función sigmoide: 1 / (1 + e^(-z))
    const probabilidad = 1 / (1 + Math.exp(-z));
    
    return probabilidad;
}

// ============================================================
// PREDICCIÓN CON RED NEURONAL
// ============================================================
async function predecirRedNeuronal(valoresEscalados) {
    const tensor = tf.tensor2d([valoresEscalados]);
    const prediccion = nnModel.predict(tensor);
    const probabilidad = (await prediccion.data())[0];
    
    tensor.dispose();
    prediccion.dispose();
    
    return probabilidad;
}

// ============================================================
// FUNCIÓN PRINCIPAL DE PREDICCIÓN
// ============================================================
async function predecir() {
    // Verificar que los modelos estén cargados
    if (!logisticParams || !nnModel) {
        alert("Los modelos aún no están cargados. Espera unos segundos.");
        return;
    }
    
    // Obtener valores del formulario
    const edad = document.getElementById('edad').value;
    const presion = document.getElementById('presion').value;
    const colesterol = document.getElementById('colesterol').value;
    const fuma = document.getElementById('fuma').value;
    const ejercicio = document.getElementById('ejercicio').value;
    
    // Validar que todos los campos estén completos
    if (!edad || !presion || !colesterol || fuma === "" || ejercicio === "") {
        alert("Por favor completa todos los campos.");
        return;
    }
    
    // Convertir a array de números
    const valores = [
        Number(edad),
        Number(presion),
        Number(colesterol),
        Number(fuma),
        Number(ejercicio)
    ];
    
    console.log("📊 Valores ingresados:", valores);
    
    // Validar que sean números válidos
    if (valores.some(v => !isFinite(v))) {
        alert("Por favor ingresa valores numéricos válidos.");
        return;
    }
    
    // Escalar los valores
    const valoresEscalados = escalar(valores);
    console.log("📐 Valores escalados:", valoresEscalados);
    
    if (valoresEscalados.some(v => !isFinite(v))) {
        alert("Error en el escalado de datos.");
        console.error("Valores escalados inválidos:", valoresEscalados);
        return;
    }
    
    try {
        // PREDICCIÓN 1: Regresión Logística
        const probLogistica = predecirLogistica(valoresEscalados);
        const resultadoLogistica = probLogistica >= 0.5 ? "RIESGO ALTO" : "RIESGO BAJO";
        
        console.log("🔵 Regresión Logística:");
        console.log("   Probabilidad:", probLogistica.toFixed(4));
        console.log("   Resultado:", resultadoLogistica);
        
        // PREDICCIÓN 2: Red Neuronal
        const probRN = await predecirRedNeuronal(valoresEscalados);
        const resultadoRN = probRN >= 0.5 ? "RIESGO ALTO" : "RIESGO BAJO";
        
        console.log("🔴 Red Neuronal:");
        console.log("   Probabilidad:", probRN.toFixed(4));
        console.log("   Resultado:", resultadoRN);
        
        // CONSENSO
        const consenso = (probLogistica >= 0.5 && probRN >= 0.5) ? "RIESGO ALTO" :
                        (probLogistica < 0.5 && probRN < 0.5) ? "RIESGO BAJO" :
                        "RIESGO MODERADO (modelos no coinciden)";
        
        // Mostrar resultados en la página
        document.getElementById('resultado-logistica').innerHTML = 
            `<span class="${resultadoLogistica === 'RIESGO ALTO' ? 'riesgo-alto' : 'riesgo-bajo'}">${resultadoLogistica}</span><br>
             Probabilidad: ${(probLogistica * 100).toFixed(2)}%`;
        
        document.getElementById('resultado-rn').innerHTML = 
            `<span class="${resultadoRN === 'RIESGO ALTO' ? 'riesgo-alto' : 'riesgo-bajo'}">${resultadoRN}</span><br>
             Probabilidad: ${(probRN * 100).toFixed(2)}%`;
        
        document.getElementById('consenso').textContent = consenso;
        
        document.getElementById('resultados').style.display = 'block';
        
        // Scroll suave hacia los resultados
        document.getElementById('resultados').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error("❌ Error durante la predicción:", error);
        alert("Error al realizar la predicción. Revisa la consola.");
    }
}