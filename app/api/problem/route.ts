import { db } from "@/lib/db";
import { auth } from "@/auth";
import { GoogleGenerativeAI } from "@google/generative-ai";

export async function POST(req: Request) {
  const session = await auth();
  console.log("Session:", session);

  if (!session?.user?.email) {
    return Response.json({ message: "Not Authenticated!" }, { status: 401 });
  }

  try {
    const { title, content } = await req.json();
    console.log("Input:", { title, content });

    // Load Gemini API
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
    const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });

    // Generate embedding
    const result = await embeddingModel.embedContent({
      content: {
        role: "user",
        parts: [{ text: `${title}\n${content}` }],
      },
    });
    const embedding = result.embedding?.values || [];

    // Store problem with embedding as string
    const newProblem = await db.problem.create({
      data: {
        title,
        content,
        authorEmail: session.user.email,
        embedding: JSON.stringify(embedding),
      },
    });

    return Response.json({ newProblem }, { status: 200 });
  } catch (error) {
    console.error("Create Problem Error:", error);
    return Response.json({ message: "Something went wrong!" }, { status: 500 });
  }
}
