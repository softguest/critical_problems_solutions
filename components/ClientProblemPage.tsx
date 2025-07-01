// 'use client';

// import { useEffect, useState } from 'react';
// import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
// import Solutions from '@/components/Solutions';
// import FormSolution from '@/components/form-solutions';
// import { Button } from '@/components/ui/button';

// interface ClientProblemPageProps {
//   problem: {
//     id: string;
//     title: string;
//     content: string;
//     // author: {
//     //   firstName: string | null;
//     //   lastName: string | null;
//     // };
//   };
//   session: any;
// }


// const ClientProblemPage = ({ problem, session }: ClientProblemPageProps) => {
//   const [refreshFlag, setRefreshFlag] = useState(false);

//   const handleSolutionSubmitted = () => {
//     setRefreshFlag(prev => !prev); // Toggle flag to trigger re-fetch in Solutions
//   };

//   return (
//     <div className="max-w-4xl py-8 mx-4">
//       <h1 className="text-3xl font-bold">{problem.title}</h1>
//       <p>
//         {/* Written by: {problem.author.firstName} <span>{problem.author.lastName}</span> */}
//       </p>
//       <div className="mt-4">{problem.content}</div>

//       {/* Always visible Solutions */}
//       <Solutions key={refreshFlag.toString()} problemId={problem.id} />

//       {/* Modal for adding a solution */}
//       <div className="mt-6">
//         <Dialog>
//           <DialogTrigger asChild>
//             <Button>Add Solution</Button>
//           </DialogTrigger>

//           <DialogContent className="max-w-xl">
//             <DialogHeader>
//               <DialogTitle>Submit a Solution</DialogTitle>
//             </DialogHeader>

//             <FormSolution
//               problemId={problem.id}
//               data={session}
//               onSuccess={handleSolutionSubmitted} // pass callback to trigger refresh
//             />
//           </DialogContent>
//         </Dialog>
//       </div>
//     </div>
//   );
// };

// export default ClientProblemPage;
