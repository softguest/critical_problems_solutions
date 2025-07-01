
import { Poppins } from "next/font/google";
import { RegisterForm } from "@/components/auth/register-form";

const font = Poppins({
  subsets: ["latin"],
  weight: ["600"]
})

const RegisterPage = () => {
  return ( 
      <RegisterForm />
  );
}
 
export default RegisterPage;