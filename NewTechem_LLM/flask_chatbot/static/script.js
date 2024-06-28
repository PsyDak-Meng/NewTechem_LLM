let isRecording = false;
let mediaRecorder;
let audioChunks = [];

const startRecordBtn = document.getElementById('start-record-btn');
const stopRecordBtn = document.getElementById('stop-record-btn');
const sendButton = document.getElementById('send-button');
const chatBox = document.getElementById('chat-box');

startRecordBtn.addEventListener('click', async () => {
    if (isRecording) return;
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
    
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const formData = new FormData();
        formData.append('audio', audioBlob);

        const response = await fetch('/chat', {
            method: 'POST',
            body: JSON.stringify({ audio: await audioBlobToBase64(audioBlob) }),
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const result = await response.json();
        updateChat(result.chat_history);
        playResponseAudio(result.response_audio);
    };
    
    isRecording = true;
    startRecordBtn.disabled = true;
    stopRecordBtn.disabled = false;
});

stopRecordBtn.addEventListener('click', () => {
    if (!isRecording) return;
    mediaRecorder.stop();
    isRecording = false;
    startRecordBtn.disabled = false;
    stopRecordBtn.disabled = true;
});

sendButton.addEventListener('click', () => {
    if (!isRecording) return;
    mediaRecorder.stop();
    isRecording = false;
    startRecordBtn.disabled = false;
    stopRecordBtn.disabled = true;
});

function updateChat(chatHistory) {
    chatBox.innerHTML = '';
    chatHistory.forEach(entry => {
        const div = document.createElement('div');
        const img = document.createElement('img');
        img.src = entry.role === 'user' ? '/static/user.gif' : '/static/bot.gif';
        img.alt = entry.role;
        img.width = 50;
        img.height = 50;
        
        const span = document.createElement('span');
        span.textContent = entry.text;

        div.appendChild(img);
        div.appendChild(span);
        chatBox.appendChild(div);
    });
    chatBox.scrollTop = chatBox.scrollHeight;
}

function playResponseAudio(audioPath) {
    const audio = new Audio(audioPath);
    audio.play();
}

async function audioBlobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}
