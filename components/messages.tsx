// ✅ Updated Messages.tsx for modern trauma support chatbot
import Markdown from "./markdown";
import { Bot, User2 } from "lucide-react";
import { Message } from "ai/react";
import React, { useEffect, useRef } from "react";
import Typewriter from "./Typewriter";

type Props = {
  messages: Message[];
  isLoading: boolean;
};

const Messages = ({ messages, isLoading }: Props) => {
  const chatRef = useRef<HTMLDivElement>(null);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (endOfMessagesRef.current) {
      endOfMessagesRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  return (
    <div
      ref={chatRef}
      className="flex flex-col w-full text-slate-700 text-left gap-4 px-2"
    >
      {messages.map((m, index) => (
        <div
          key={index}
          className={`p-4 rounded-2xl shadow-sm relative max-w-[85%] md:max-w-[70%] whitespace-pre-wrap ${
            m.role === "user" ? "bg-white ml-auto border border-indigo-200" : "bg-indigo-100 ml-2"
          }`}
        >
          {m.role === "assistant" && isLoading && index === messages.length - 1 ? (
            <Typewriter text={m.content} />
          ) : (
            <Markdown text={m.content} />
          )}

          {m.role === "user" ? (
            <User2 className="absolute -right-10 top-2 border bg-white rounded-full p-1 shadow-md" />
          ) : (
            <Bot
              className={`absolute -left-10 top-2 border bg-indigo-50 rounded-full p-1 shadow-md stroke-indigo-600 ${
                isLoading && index === messages.length - 1 ? "animate-pulse" : ""
              }`}
            />
          )}
        </div>
      ))}
      <div ref={endOfMessagesRef} />
    </div>
  );
};

export default Messages;