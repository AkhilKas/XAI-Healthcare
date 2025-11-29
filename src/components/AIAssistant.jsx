import React, { useState } from 'react';
import { Send } from 'lucide-react';

export const AIAssistant = () => {
  const [messages, setMessages] = useState([
    { type: 'ai', content: 'Hello! How can I help you analyze this patient\'s motion data?' }
  ]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    
    setMessages([...messages, { type: 'user', content: input }]);
    setInput('');
    
    // Simulate AI response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        type: 'ai',
        content: 'Patient shows improving ROM with moderate scapular compensation.'
      }]);
    }, 1000);
  };

  return (
    <div className="flex-1 flex flex-col p-4 min-h-0">
      <div className="bg-blue-600 text-white px-4 py-2 rounded-t-lg flex items-center gap-2">
        <div className="w-2 h-2 bg-white rounded-full"></div>
        <span className="font-medium text-sm">AI Assistant</span>
      </div>
      
      <div className="flex-1 bg-gray-50 border border-gray-200 rounded-b-lg flex flex-col min-h-0">
        <div className="flex-1 p-3 overflow-y-auto space-y-3">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] px-3 py-2 rounded-lg text-xs ${
                msg.type === 'user' 
                  ? 'bg-blue-600 text-white rounded-br-none' 
                  : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none'
              }`}>
                {msg.content}
              </div>
            </div>
          ))}
        </div>
        
        <div className="p-2 border-t border-gray-200 bg-white">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask a question..."
              className="flex-1 px-3 py-2 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
              onClick={handleSend}
              className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
