'use client';

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';

type Message = {
  role: 'user' | 'ai';
  content: string;
};

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'ai',
      content:
        "👋 Hello! I'm your AI solution assistant.\nDescribe any problem you're facing, and I'll help find relevant solutions from our knowledge base.",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // const sendMessage = async () => {
  //   if (!input.trim()) return;

  //   const newMessages = [...messages, { role: 'user', content: input }];
  //   setMessages(newMessages);
  //   setInput('');
  //   setLoading(true);

  //   try {
  //     const res = await fetch('/api/chat/ai', {
  //       method: 'POST',
  //       headers: { 'Content-Type': 'application/json' },
  //       body: JSON.stringify({ message: input }),
  //     });

  //     const data = await res.json();

  //     // AI reply message
  //     const aiReplyMessage = {
  //       role: 'ai',
  //       content: data.reply || "🤖 I'm thinking... Please try again.",
  //     };

  //     // Problem match summary
  //     const problemMatchMessage =
  //       data.problems && data.problems.length > 0
  //         ? {
  //             role: 'ai',
  //             content: data.problems
  //               .map(
  //                 (p: any) =>
  //                   `🧠 **Matched Problem:** ${p.title}\n📄 ${p.content}\n✅ **Solutions:**\n${p.solutions
  //                     .map((s: any) => `- ${s.content}`)
  //                     .join('\n')}`
  //               )
  //               .join('\n\n'),
  //           }
  //         : {
  //             role: 'ai',
  //             content:
  //               "❗ I couldn't find any exact matches. Could you describe your problem in more detail?",
  //           };

  //     setMessages((prev) => [...prev, aiReplyMessage, problemMatchMessage]);
  //   } catch (err) {
  //     console.error('[CHAT ERROR]', err);
  //     setMessages((prev) => [
  //       ...prev,
  //       {
  //         role: 'ai',
  //         content: '⚠️ Sorry, something went wrong. Please try again later.',
  //       },
  //     ]);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat/ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      const data = await res.json();
      setMessages((prev) => [...prev, { role: 'ai', content: data.reply }]);
    } catch (err) {
      console.error('[CHAT ERROR]', err);
      setMessages((prev) => [
        ...prev,
        { role: 'ai', content: 'Sorry, something went wrong.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="max-w-3xl mx-auto mt-8">
      <CardContent className="p-6 space-y-4">
        <ScrollArea className="h-[400px] pr-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`${
                msg.role === 'user' ? 'text-right' : 'text-left'
              } my-2`}
            >
              <div
                className={`inline-block whitespace-pre-wrap px-4 py-2 rounded-lg ${
                  msg.role === 'user'
                    ? 'bg-blue-100 text-blue-900'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}
        </ScrollArea>

        <div className="flex items-center gap-2">
          <Input
            placeholder="Describe your problem or ask for specific solutions..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          />
          <Button onClick={sendMessage} disabled={loading}>
            {loading ? 'Sending...' : 'Send'}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Powered by AI • Your feedback helps us improve responses
        </p>
      </CardContent>
    </Card>
  );
}
