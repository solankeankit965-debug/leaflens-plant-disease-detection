import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'LeafLens — AI Plant Disease Detection',
  description: 'Scan a leaf, identify plant disease, and receive practical treatment guidance.',
  openGraph: {
    title: 'LeafLens — AI Plant Disease Detection',
    description: 'Scan a leaf and turn visual symptoms into a clear crop-care plan.',
    type: 'website',
    images: [{ url: '/og.png', width: 1731, height: 909, alt: 'LeafLens AI plant disease detection' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LeafLens — AI Plant Disease Detection',
    description: 'Scan a leaf and turn visual symptoms into a clear crop-care plan.',
    images: ['/og.png'],
  },
  icons: { icon: '/favicon.svg' },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
