import {db} from "@/lib/db";
import {auth } from "@/auth"

export async function POST(req: Request) {
  const session = await auth();

  try {
    if(!session?.user?.email) {
      return Response.json({ message: 'Not Authenticated!' }, { status: 401 })
    }

    const { problemId, title, content } = await req.json();
    const newProblem = await db.solution.create({
      data: {
        problemId, title, content, authorEmail: session.user.email
      }
    })
    return Response.json({newProblem}, { status: 200})

  } catch(error) {
    return Response.json({ message: 'Something went wrong!'}, { status: 500 })
  }
}