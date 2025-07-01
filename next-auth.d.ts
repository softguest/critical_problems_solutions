import { UserRole } from "@prisma/client";
import NextAuth, { type DefaultSession } from "next-auth";

export type ExtendedUser = DefaultSession["user"] & {
  profileImage:              string;
  firstName:                 string;
  middleName:                string;
  lastName:                  string;
  gender:                    string;
  dateOfBirth:               string;
  telePhone:                 string;

  cityName:                   string;
  country:                    string;
  pinCode:                    string;

  email:                    string;
  role:                     UserRole;
  isTwoFactorEnabled:       boolean;
  isOAuth:                  boolean;
  bio:                      string;

  typeOfIdentificationCard: string;
  identificationCardNumber: string;
  identificationFile: string;
  
};

declare module "next-auth" {
  interface Session {
    user: ExtendedUser;
  }
}
