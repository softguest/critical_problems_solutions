import { Poppins } from "next/font/google";
import { LoginForm } from "@/components/auth/login-form";

const font = Poppins({
  subsets: ["latin"],
  weight: ["600"]
})

export default function LoginPage() {
  return (
    <div className="h-full flex items-center justify-center  bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-orange-50 to-orange-300">
      <LoginForm />
    </div>
  )
}
