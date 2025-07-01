/*
  Warnings:

  - You are about to drop the column `userId` on the `problem` table. All the data in the column will be lost.
  - You are about to drop the column `userId` on the `solution` table. All the data in the column will be lost.
  - Added the required column `title` to the `Solution` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE `problem` DROP FOREIGN KEY `Problem_userId_fkey`;

-- DropForeignKey
ALTER TABLE `solution` DROP FOREIGN KEY `Solution_problemId_fkey`;

-- DropForeignKey
ALTER TABLE `solution` DROP FOREIGN KEY `Solution_userId_fkey`;

-- AlterTable
ALTER TABLE `problem` DROP COLUMN `userId`,
    ADD COLUMN `authorEmail` VARCHAR(191) NULL,
    MODIFY `content` VARCHAR(191) NOT NULL;

-- AlterTable
ALTER TABLE `solution` DROP COLUMN `userId`,
    ADD COLUMN `authorEmail` VARCHAR(191) NULL,
    ADD COLUMN `title` VARCHAR(191) NOT NULL,
    MODIFY `content` VARCHAR(191) NOT NULL,
    MODIFY `problemId` VARCHAR(191) NULL;

-- AddForeignKey
ALTER TABLE `Problem` ADD CONSTRAINT `Problem_authorEmail_fkey` FOREIGN KEY (`authorEmail`) REFERENCES `User`(`email`) ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE `Solution` ADD CONSTRAINT `Solution_authorEmail_fkey` FOREIGN KEY (`authorEmail`) REFERENCES `User`(`email`) ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE `Solution` ADD CONSTRAINT `Solution_problemId_fkey` FOREIGN KEY (`problemId`) REFERENCES `Problem`(`id`) ON DELETE SET NULL ON UPDATE CASCADE;
