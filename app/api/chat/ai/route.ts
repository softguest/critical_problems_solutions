// File: /app/api/chat/ai/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { cosineSimilarity } from '@/lib/similarity';

export async function POST(req: NextRequest) {
  try {
    const { message } = await req.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json({ error: 'Invalid message input.' }, { status: 400 });
    }

    // Generate embedding for user query
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: 'embedding-001',
      apiKey: process.env.GEMINI_API_KEY!,
    });

    let userVector: number[] = [];
    try {
      userVector = await embeddings.embedQuery(message);
    } catch (embeddingErr) {
      console.error('[Embedding Error]', embeddingErr);
      return NextResponse.json({ error: 'Failed to generate embedding.' }, { status: 500 });
    }

    // Fetch all problems (with solutions and embeddings)
    const problems = await db.problem.findMany({
      include: { solutions: true },
    });

    // Filter out problems without valid embeddings
    const scoredProblems = problems
      .filter((p) => Array.isArray(p.embedding))
      .map((p) => {
        const similarity = cosineSimilarity(userVector, p.embedding as number[]);
        return { ...p, similarity };
      });

    // If no valid embeddings found
    if (scoredProblems.length === 0) {
      return NextResponse.json({
        matched: false,
        problems: [],
        suggestions: [],
        message: "No valid problem embeddings found in the database.",
      });
    }

    // Sort problems by similarity score
    scoredProblems.sort((a, b) => b.similarity - a.similarity);

    const bestMatch = scoredProblems[0];
    const suggestions = scoredProblems.slice(1, 4); // top 3 similar alternatives

    // If high confidence match
    if (bestMatch && bestMatch.similarity >= 0.9) {
      return NextResponse.json({
        matched: true,
        problems: [bestMatch],
        suggestions,
      });
    } else {
      return NextResponse.json({
        matched: false,
        suggestions,
        message: "Couldn't confidently identify your problem, but here are some similar ones.",
      });
    }
  } catch (err) {
    console.error('[CHAT AI ERROR]', err);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
