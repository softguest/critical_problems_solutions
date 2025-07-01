"use client";

import { useChat } from "ai/react";
import { useRef, useEffect } from "react";
import {
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { ScrollArea } from "./ui/scroll-area";
import Messages from "./messages";
import InputForm from "./inputForm";

const ChatSection = () => {
  const scrollRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    stop,
  } = useChat({
    api: "api/gemini",
  });

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (  
    <div className="mx-4 my-4 flex justify-center">
      <div className="w-full max-w-3xl bg-white border border-gray-300 rounded-xl shadow-lg flex flex-col md:mt-[-300px]">
        
        <CardHeader className="border-b border-gray-200 px-6 py-4 bg-white rounded-xl">
          <CardTitle className="text-xl font-semibold text-blue-800">
            💬 Ask Anything About Your Problem.
          </CardTitle>
        </CardHeader>

        <CardContent className="px-6 pt-4 pb-2 h-[500px] overflow-y-auto">
          <ScrollArea className="h-full pr-3">
            <Messages messages={messages} isLoading={isLoading} />
            <div ref={scrollRef} />
          </ScrollArea>
        </CardContent>

        <CardFooter className="border-t border-gray-200 px-6 py-4 bg-white rounded-b-xl">
          <InputForm
            input={input}
            handleInputChange={handleInputChange}
            handleSubmit={handleSubmit}
            isLoading={isLoading}
            stop={stop}
          />
        </CardFooter>

      </div>
    </div>
  );
};

export default ChatSection;
