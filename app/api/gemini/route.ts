import { Message } from "ai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { initailMessage } from "@/lib/data";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

export const runtime = "edge";

const buildGeminiPrompt = (messages: Message[]): any => {
  const geminiMessages = [
    {
      role: "user",
      parts: initailMessage.contents, 
    },
    ...messages.map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    })),
  ];
  return geminiMessages;
};

export async function POST(request: Request) {
  try {
    const { messages } = await request.json();
    const geminiPrompt = buildGeminiPrompt(messages);

    const geminiStream = await model.generateContentStream({
      contents: geminiPrompt,
      generationConfig: {
        temperature: 0.7,
      },
    }); 

    const stream = new ReadableStream({
      async start(controller) {
        for await (const chunk of geminiStream.stream) {
          controller.enqueue(new TextEncoder().encode(chunk.text()));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain",
        "Transfer-Encoding": "chunked",
      },
    });
  } catch (error) {
    console.error("Gemini API Error:", error);
    return new Response(
      JSON.stringify({ error: "Failed to generate response from Gemini API" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}