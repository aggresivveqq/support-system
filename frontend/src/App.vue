<template>
  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Магистерская система AIOps</p>
        <h1>Интеллектуальный анализ логов контейнерной инфраструктуры</h1>
        <p class="subtitle">
          ML сначала классифицирует события по уровню критичности. Решения ИИ загружаются
          отдельным шагом — пользователь видит прогресс, а не ждёт единого ответа.
        </p>
      </div>
      <div class="status-panel">
        <span :class="['pulse', health.openrouter_enabled ? 'ok' : 'warn']"></span>
        <div>
          <strong>{{ health.openrouter_enabled ? 'OpenRouter подключён' : 'OpenRouter не настроен' }}</strong>
          <small>ML-модель: {{ health.model_exists ? 'загружена' : 'не найдена' }}</small>
        </div>
      </div>
    </section>

    <section class="workspace">
      <div class="input-pane">
        <div class="pane-head">
          <h2>Входные логи</h2>
          <label class="toggle">
            <input type="checkbox" v-model="useLlm" />
            <span>Внешний LLM через OpenRouter</span>
          </label>
        </div>
        <textarea v-model="logs" spellcheck="false" placeholder="Вставьте лог-строки, по одной на строку…"></textarea>
        <div class="actions">
          <button @click="analyze" :disabled="mlLoading || aiLoading || !logs.trim()">
            <span v-if="mlLoading" class="spinner"></span>
            {{ mlLoading ? 'ML классифицирует…' : 'Анализировать' }}
          </button>
          <button class="ghost" @click="loadExample" :disabled="mlLoading || aiLoading">Пример инцидента</button>
        </div>
      </div>

      <div class="decision-pane">
        <h2>ИИ-анализ и поддержка решений</h2>

        <div v-if="mlLoading" class="empty progress-box">
          <span class="spinner"></span>
          ML классифицирует логи по уровню критичности…
        </div>

        <div v-else-if="result" class="decision-content">
          <div class="risk-row">
            <span :class="['badge', result.summary.max_severity]">{{ result.summary.max_severity }}</span>
            <span>{{ result.summary.total }} событий обработано</span>
          </div>

          <div v-if="aiLoading" class="ai-loading">
            <span class="spinner"></span>
            <div>
              <strong>Результат ML готов.</strong>
              <small>Генерируется объяснение ИИ и рекомендации…</small>
            </div>
          </div>

          <template v-else-if="result.decision_support?.analysis">
            <div class="llm-text">{{ truncatedAnalysis }}</div>
            <button
              v-if="analysisOverflow"
              class="ghost compact"
              @click="showFullAnalysis = !showFullAnalysis"
            >
              {{ showFullAnalysis ? 'Свернуть' : `Показать полностью (${analysisLineCount} строк)` }}
            </button>
          </template>

          <template v-else-if="result.decision_support">
            <p class="hypothesis">{{ result.decision_support.main_hypothesis }}</p>
            <ul>
              <li v-for="action in result.decision_support.recommended_actions" :key="action">{{ action }}</li>
            </ul>
          </template>

          <small v-if="result.decision_support" class="provider">
            Источник: {{ result.decision_support.provider }}
          </small>
        </div>

        <div v-else class="empty">
          Вставьте логи контейнера, Kubernetes-ноды, Docker daemon или приложения
          и запустите анализ.
        </div>
      </div>
    </section>

    <section class="metrics" v-if="result">
      <article v-for="label in labels" :key="label" :class="['metric', label]">
        <span>{{ severityRu[label] }}</span>
        <strong>{{ result.summary.distribution[label] || 0 }}</strong>
      </article>
    </section>

    <section class="table-section" v-if="result">
      <h2>ML-предсказания ({{ result.predictions.length }} строк)</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Критичность</th>
              <th>Уверенность</th>
              <th>Источник</th>
              <th>Сообщение лога</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in result.predictions" :key="item.line_no">
              <td>{{ item.line_no }}</td>
              <td><span :class="['badge', item.severity]">{{ item.severity }}</span></td>
              <td>
                <div class="conf-bar">
                  <span class="confidence-label">{{ Math.round(item.confidence * 100) }}%</span>
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fill" :style="{ width: Math.round(item.confidence * 100) + '%' }"></div>
                  </div>
                </div>
              </td>
              <td class="source-cell">{{ item.source }}</td>
              <td class="message" :title="item.message">{{ item.message }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="dataset" v-if="dataset">
      <div class="panel-row">
        <div>
          <h2>Данные LogHub</h2>
          <p>{{ dataset.rows }} строк · источники: {{ dataset.sources.join(', ') }}</p>
        </div>
        <div class="chips">
          <span v-for="(value, key) in dataset.class_distribution" :key="key">{{ key }}: {{ value }}</span>
        </div>
      </div>
    </section>

    <section class="storage" v-if="storage">
      <div class="panel-row">
        <div>
          <h2>SQLite storage</h2>
          <p>{{ storage.analysis_runs }} запусков анализа · {{ storage.stored_log_predictions }} сохранённых предсказаний</p>
        </div>
        <div class="chips">
          <span v-for="label in labels" :key="label">{{ severityRu[label] }}: {{ storage.severity_distribution[label] || 0 }}</span>
        </div>
      </div>
    </section>

    <section class="history" v-if="history.length">
      <h2>История анализов</h2>
      <div class="history-list">
        <button
          v-for="item in history"
          :key="item.id"
          class="history-item"
          @click="loadHistoryItem(item.id)"
        >
          <span :class="['badge', item.max_severity]">{{ item.max_severity }}</span>
          <strong>#{{ item.id }}</strong>
          <span>{{ item.total }} строк</span>
          <small>{{ item.created_at }}</small>
        </button>
      </div>
    </section>
  </main>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'

const ANALYSIS_MAX_LINES = 30
const labels = ['info', 'warning', 'alert', 'disaster']
const severityRu = { info: 'инфо', warning: 'предупр.', alert: 'тревога', disaster: 'авария' }

const useLlm = ref(true)
const mlLoading = ref(false)
const aiLoading = ref(false)
const result = ref(null)
const dataset = ref(null)
const storage = ref(null)
const history = ref([])
const health = ref({ model_exists: false, openrouter_enabled: false })
const showFullAnalysis = ref(false)

const logs = ref(`INFO kubelet Started container api-gateway in pod production/api-gateway-74ff
WARNING container runtime reports high memory usage threshold exceeded
ERROR api-gateway failed to connect to postgres: timeout after 30s
FATAL payment-service kernel panic: unrecoverable corruption detected`)

const analysisText = computed(() => result.value?.decision_support?.analysis ?? '')
const analysisLines = computed(() => analysisText.value.split('\n'))
const analysisLineCount = computed(() => analysisLines.value.length)
const analysisOverflow = computed(() => analysisLineCount.value > ANALYSIS_MAX_LINES)
const truncatedAnalysis = computed(() => {
  if (!analysisOverflow.value || showFullAnalysis.value) return analysisText.value
  return analysisLines.value.slice(0, ANALYSIS_MAX_LINES).join('\n') + '\n...'
})

async function analyze() {
  mlLoading.value = true
  aiLoading.value = false
  showFullAnalysis.value = false
  result.value = null

  try {
    const classifyResponse = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ logs: logs.value, use_llm: useLlm.value })
    })
    result.value = {
      ...(await classifyResponse.json()),
      decision_support: null
    }
  } finally {
    mlLoading.value = false
  }

  aiLoading.value = true
  try {
    const decisionResponse = await fetch('/api/decision', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ logs: logs.value, use_llm: useLlm.value })
    })
    const decision = await decisionResponse.json()
    result.value = {
      ...result.value,
      summary: { ...result.value.summary, ...decision.summary },
      decision_support: decision.decision_support
    }
    await refreshStorage()
  } finally {
    aiLoading.value = false
  }
}

async function refreshStorage() {
  storage.value = await (await fetch('/api/storage')).json()
  history.value = (await (await fetch('/api/history?limit=8')).json()).items
}

async function loadHistoryItem(id) {
  result.value = await (await fetch(`/api/history/${id}`)).json()
  logs.value = result.value.predictions.map((item) => item.message).join('\n')
  showFullAnalysis.value = false
}

function loadExample() {
  logs.value = `INFO docker Container nginx started with image nginx:latest
WARNING kubelet Back-off restarting failed container worker
ERROR postgres connection refused for service billing-api
ERROR billing-api request failed: timeout while calling payment-service
FATAL node kernel panic: out of memory and system down`
}

onMounted(async () => {
  health.value = await (await fetch('/api/health')).json()
  dataset.value = await (await fetch('/api/dataset')).json()
  await refreshStorage()
})
</script>
