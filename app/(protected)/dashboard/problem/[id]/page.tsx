import { FC} from 'react';
import { auth } from '@/auth';
import Solutions from '@/components/Solutions';
import FormSolution from '@/components/form-solutions';
import { db } from '@/lib/db';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

interface ProblemDetailPageProps {
  params: {
    id: string;
  };
}

const ProblemDetailPage: FC<ProblemDetailPageProps> = async ({ params }) => {
  const session = await auth();
  const problem = await db.problem.findFirst({
    where: {
      id: params.id,
    },
    include: {
      author: true,
    },
  });

  return (
    <div className="max-w-4xl py-4 mx-4">
      <div className='mb-6'>
        <Dialog>
          <DialogTrigger asChild>
            <button className="bg-primary text-white px-4 py-2 rounded-md hover:bg-primary/80 transition">
              Add Solution
            </button>
          </DialogTrigger>

          {/* Modal Content */}
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Submit a Solution</DialogTitle>
            </DialogHeader>

            <div className="space-y-6">
              <FormSolution problemId={params.id} data={session} />
            </div>
          </DialogContent>
        </Dialog>
      </div>
      <h1 className="text-3xl font-bold">{problem?.title}</h1>
      <p>
        Written by: {problem?.author?.firstName}{' '}
        <span>{problem?.author?.lastName}</span>
      </p>
      <div className="mt-4">{problem?.content}</div>

      <Solutions problemId={params.id} />

      <div className="mt-6">
        {/* Modal Trigger */}
        
      </div>
    </div>
  );
};

export default ProblemDetailPage;
