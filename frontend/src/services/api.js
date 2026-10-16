/**
 * Thin API client for the FraudShield AI backend.
 *
 * The base URL defaults to `/api`, which Vite proxies to the FastAPI server in
 * development. Override with VITE_API_BASE_URL for other environments.
 */
const BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

function describeDetail(detail, fallback) {
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const loc = Array.isArray(d.loc) ? d.loc.filter((p) => p !== 'body').join(' -> ') : ''
        return loc ? `${loc}: ${d.msg}` : d.msg
      })
      .join('; ')
  }
  if (typeof detail === 'object') {
    if (detail.message) return detail.message
    return JSON.stringify(detail)
  }
  return fallback
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, options)
  } catch {
    throw new ApiError('Could not reach the backend. Is the FastAPI server running?', 0, null)
  }

  const text = await response.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }

  if (!response.ok) {
    throw new ApiError(
      describeDetail(body?.detail, `Request failed with status ${response.status}`),
      response.status,
      body,
    )
  }
  return body
}

export const api = {
  baseUrl: BASE_URL,

  health: () => request('/health'),

  /**
   * Upload a transaction CSV and get every row classified.
   * Uses XHR so upload progress can be shown for large (~144 MB) files.
   */
  analyze: (file, model, onProgress) => {
    const form = new FormData()
    form.append('file', file)
    if (model) form.append('model', model)

    if (!onProgress) {
      return request('/analyze', { method: 'POST', body: form })
    }

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE_URL}/analyze`)
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress(Math.round((event.loaded / event.total) * 100))
        }
      }
      xhr.onload = () => {
        let body = null
        try {
          body = JSON.parse(xhr.responseText)
        } catch {
          body = null
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body)
        else {
          reject(
            new ApiError(
              describeDetail(body?.detail, `Upload failed with status ${xhr.status}`),
              xhr.status,
              body,
            ),
          )
        }
      }
      xhr.onerror = () => reject(new ApiError('Upload failed - is the backend running?', 0, null))
      xhr.send(form)
    })
  },
}

export default api
