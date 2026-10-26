-- CreateTable (IF NOT EXISTS: access.py may already have created it at app startup)
CREATE TABLE IF NOT EXISTS "AllowedUser" (
    "email" TEXT NOT NULL,
    "addedBy" TEXT,
    "createdAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AllowedUser_pkey" PRIMARY KEY ("email")
);
