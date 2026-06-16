import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

export async function fetchPdfList() {
    const res = await axios.get(`${API_BASE}/pdf-list`);
    return res.data.pdf_files || [];
}

export async function deletePdf(filename) {
    return axios.delete(`${API_BASE}/delete-pdf`, { data: { filename } });
}
